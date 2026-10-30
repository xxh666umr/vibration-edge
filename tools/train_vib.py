"""工业振动（刀具磨损 4 分类）MCU 友好模型训练脚本。

数据：SDU-QIT x 轴，1024 点 @10.24 kHz，4 类（空闲/初期/稳定/加速磨损）。
两种划分：
  --split random  论文 Table I 分层随机划分（与 TWM-Net 98.31% 可比）
  --split block   按时间块划分（前 70% 训练 / 中 20% 验证 / 后 10% 测试，更严格）

模型 VibNet：第一层大 stride 下采样，控制 MAC 数，便于 int8 部署到 STM32F103。
用法：
    python tools/train_vib.py --split random --epochs 60
    python tools/train_vib.py --split block  --epochs 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

CLASSES = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]
DATA_DIR = r"C:\Users\lenovo\Desktop\工业语义通信Demo项目\data\Toolweardata"
# 论文 Table I：每类 train / val 数量（test = 总数 - train - val）
PAPER_TRAIN = [7816, 4613, 39968, 10761]
PAPER_VAL = [2233, 1317, 11419, 3074]


def load_axis(axes="x"):
    """axes: "x" / "xy" / "xyz"，返回 (N, C, 1024) 或 (N, 1024) 与长度 N 的标签。"""
    xs_all, y = [], None
    for ax in axes:
        per_ax, ys_ax = [], []
        for c in range(4):
            a = np.load(os.path.join(DATA_DIR, f"trainData_{ax}_{c}.npy"))
            a = np.asarray(a, dtype=np.float32)
            per_ax.append(a)
            ys_ax.append(np.full(len(a), c, dtype=np.int64))
        xs_all.append(np.concatenate(per_ax))
        if y is None:
            y = np.concatenate(ys_ax)
    X = np.stack(xs_all, axis=1) if len(xs_all) > 1 else xs_all[0]
    assert len(X) == len(y), (len(X), len(y))
    return X, y


def split_random(y, seed=2024):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(4):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nt, nv = PAPER_TRAIN[c], PAPER_VAL[c]
        tr.extend(idx[:nt])
        va.extend(idx[nt:nt + nv])
        te.extend(idx[nt + nv:])
    return np.asarray(tr), np.asarray(va), np.asarray(te)


def split_block(y):
    """按时间块切分：每类前 70% 训练、中 20% 验证、后 10% 测试（保持时间顺序）。"""
    tr, va, te = [], [], []
    for c in range(4):
        idx = np.where(y == c)[0]  # 文件内顺序即时间顺序
        n = len(idx)
        n_tr = int(round(0.7 * n))
        n_va = int(round(0.2 * n))
        tr.extend(idx[:n_tr])
        va.extend(idx[n_tr:n_tr + n_va])
        te.extend(idx[n_tr + n_va:])
    return np.asarray(tr), np.asarray(va), np.asarray(te)


class VibNet(nn.Module):
    """MCU 友好 1D-CNN：首层大 stride 下采样，参数 ~8k，MAC ~75k。"""

    def __init__(self, c1=16, c2=32, c3=48, k1=16, s1=16, nc=4, in_c=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, c1, k1, s1)
        self.conv2 = nn.Conv1d(c1, c2, 5, 2)
        self.conv3 = nn.Conv1d(c2, c3, 3, 1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(c3, nc)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 1024 -> 63 -> 31
        x = self.pool(torch.relu(self.conv2(x)))   # 31 -> 14 -> 7
        x = torch.relu(self.conv3(x))              # 7
        x = x.mean(dim=2)                          # GAP
        return self.fc(x)


def macs(c1, c2, c3, k1, s1, in_len=1024, in_c=1):
    l1 = (in_len - k1) // s1 + 1
    l1p = l1 // 2
    l2 = (l1p - 5) // 2 + 1
    l2p = l2 // 2
    l3 = l2p
    return (c1 * l1 * in_c * k1) + (c2 * l2 * c1 * 5) + (c3 * l3 * c2 * 3)


def augment(x, gen):
    """轻度增强：随机循环时移 + ±5% 幅度缩放（保幅类信息，不做逐窗归一化）。"""
    if float(torch.rand(1, device=x.device, generator=gen)) < 0.5:
        sh = int(torch.randint(-64, 65, (1,), device=x.device, generator=gen).item())
        x = torch.roll(x, sh, dims=2)
    if float(torch.rand(1, device=x.device, generator=gen)) < 0.3:
        g = 1.0 + (float(torch.rand(1, device=x.device, generator=gen).item()) - 0.5) * 0.1
        x = x * g
    return x


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "accuracy": float(100 * tp.sum() / cm.sum()),
        "precision": float(prec.mean() * 100),
        "recall": float(rec.mean() * 100),
        "f1_macro": float(f1.mean() * 100),
        "per_class_f1": {CLASSES[i]: float(f1[i] * 100) for i in range(4)},
        "per_class_recall": {CLASSES[i]: float(rec[i] * 100) for i in range(4)},
        "confusion": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, X, y, dev, batch=1024):
    model.eval()
    cm = np.zeros((4, 4), np.int64)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        if isinstance(xb, np.ndarray):
            xb = torch.from_numpy(xb)
        xb = xb.to(dev)
        pred = model(xb).argmax(1).cpu().numpy()
        for t, p in zip(y[i:i + batch], pred):
            cm[t, p] += 1
    return metrics_from_cm(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="x", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"])
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--c3", type=int, default=48)
    ap.add_argument("--k1", type=int, default=16)
    ap.add_argument("--s1", type=int, default=16)
    ap.add_argument("--class-weight", action="store_true", help="按类别频率加权 loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    in_c = len(args.axes)
    X, y = load_axis(args.axes)
    print(f"[data] {X.shape}, per-class {np.bincount(y, minlength=4).tolist()}", flush=True)
    tr, va, te = (split_random(y) if args.split == "random" else split_block(y))
    print(f"[split={args.split}] train {len(tr)} val {len(va)} test {len(te)}, "
          f"per-class test {np.bincount(y[te], minlength=4).tolist()}", flush=True)

    if in_c == 1:
        Xtr = torch.from_numpy(X[tr][:, None, :]).to(dev)
        Xva = torch.from_numpy(X[va][:, None, :]).to(dev)
        Xte = torch.from_numpy(X[te][:, None, :]).to(dev)
    else:
        Xtr = torch.from_numpy(X[tr]).to(dev)
        Xva = torch.from_numpy(X[va]).to(dev)
        Xte = torch.from_numpy(X[te]).to(dev)
    ytr = torch.from_numpy(y[tr]).to(dev)

    torch.manual_seed(args.seed)
    model = VibNet(args.c1, args.c2, args.c3, args.k1, args.s1, in_c=in_c).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    mc = macs(args.c1, args.c2, args.c3, args.k1, args.s1, in_c=in_c)
    print(f"[model] params={n_par}  MAC~{mc}", flush=True)

    if args.class_weight:
        freq = np.bincount(y[tr], minlength=4).astype(np.float64)
        w = torch.tensor((len(tr) / (4 * freq)), dtype=torch.float32, device=dev)
        print(f"[class weight] {w.cpu().numpy().round(3).tolist()}", flush=True)
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    best_va, best_state, bad = 0.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot, corr = 0.0, 0
        for i in range(0, len(tr), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(Xtr[idx], gen)
            yb = ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            corr += (out.argmax(1) == yb).sum().item()
        sched.step()
        vm = evaluate(model, Xva, y[va], dev)
        if vm["accuracy"] > best_va:
            best_va = vm["accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}] loss={tot/len(tr):.4f} train_acc={100*corr/len(tr):.2f}% "
                  f"val_acc={vm['accuracy']:.2f}% val_f1={vm['f1_macro']:.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= 15 and ep > 25:
            print(f"[early stop] ep {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    res = {
        "model": "VibNet", "axes": args.axes, "split": args.split,
        "params": int(n_par), "macs": int(mc),
        "config": {k: getattr(args, k) for k in ("c1", "c2", "c3", "k1", "s1")},
        "class_weight": args.class_weight, "epochs": args.epochs,
        "best_val_acc": best_va,
    }
    for name, Xe, ye in (("test", Xte, y[te]), ("val", Xva, y[va])):
        res[name] = evaluate(model, Xe, ye, dev)
        print(f"[{name}] acc={res[name]['accuracy']:.2f}% f1={res[name]['f1_macro']:.2f}% "
              f"per-class F1={ {k: round(v,1) for k,v in res[name]['per_class_f1'].items()} }",
              flush=True)
    res["data_per_class_test"] = np.bincount(y[te], minlength=4).tolist()
    res["time_seconds"] = time.time() - t0

    ckpt = os.path.join(args.out, "vibnet.pt")
    torch.save({"state_dict": best_state, "config": res["config"],
                "classes": CLASSES, "axes": args.axes, "split": args.split}, ckpt)
    out_json = os.path.join(args.out, f"metrics_{args.split}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[saved] {ckpt}\n[saved] {out_json}", flush=True)


if __name__ == "__main__":
    main()
