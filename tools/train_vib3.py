r"""三轴（x/y/z 同时输入）两级方案：RMS 门限 + 3 类磨损 CNN。

与单轴版本的区别：
- 输入 (N, 3, 1024)，三轴联合逐窗归一化（同除一个 RMS）；
- 门限用三轴联合 AC-RMS；
- 模型 conv1 输入通道 = 3，其余结构不变（参数量略增）。

用法：python tools/train_vib3.py --split random --epochs 80 --out models\gate3_xyz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vib import CLASSES, VibNet, augment, metrics_from_cm, split_block, split_random  # noqa: E402

WEAR = ["initial_wear", "steady_wear", "accelerating_wear"]
DATA_DIR = r"C:\Users\lenovo\Desktop\工业语义通信Demo项目\data\Toolweardata"


def load_xyz():
    xs, y = [], None
    for ax in ("x", "y", "z"):
        per, ys = [], []
        for c in range(4):
            a = np.asarray(np.load(os.path.join(DATA_DIR, f"trainData_{ax}_{c}.npy")), np.float32)
            per.append(a)
            ys.append(np.full(len(a), c, np.int64))
        xs.append(np.concatenate(per))
        if y is None:
            y = np.concatenate(ys)
    return np.stack(xs, axis=1), y  # (N,3,1024)


def norm3(X, rms, clip=6.0):
    return np.clip(X / np.maximum(rms[:, None, None], 1e-9), -clip, clip).astype(np.float32)


@torch.no_grad()
def predict(model, X, dev, batch=512):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(dev)
        outs.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--c3", type=int, default=48)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--s1", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    X, y = load_xyz()
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    print(f"[data] {X.shape}, per-class {np.bincount(y, minlength=4).tolist()}", flush=True)
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True

    thr = float(rms[(y == 0) & m_tr].max()) * 3.0
    print(f"[gate] idling rms max(train)={rms[(y==0)&m_tr].max():.5f} threshold={thr:.5f}", flush=True)

    wtr = np.where(m_tr & (y > 0))[0]
    wva = np.where((np.zeros(len(y), bool) | np.isin(np.arange(len(y)), va)) & (y > 0))[0]
    wte = np.where(m_te & (y > 0))[0]
    Xtr = norm3(X[wtr], rms[wtr])
    ytr = y[wtr] - 1
    Xva = norm3(X[wva], rms[wva])
    yva = y[wva] - 1
    Xte = norm3(X[wte], rms[wte])
    yte = y[wte] - 1

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    torch.manual_seed(args.seed)
    model = VibNet(args.c1, args.c2, args.c3, args.k1, args.s1, nc=3).to(dev)
    # 把 conv1 的输入通道改成 3
    model.conv1 = nn.Conv1d(3, args.c1, args.k1, args.s1).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_par}", flush=True)

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    best, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = crit(model(augment(Xtr_t[idx], gen)), ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        va_acc = 100.0 * float((predict(model, Xva, dev) == yva).mean())
        if va_acc > best:
            best = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}] loss={tot/len(Xtr):.4f} val_wear_acc={va_acc:.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    pred = predict(model, Xte, dev)
    wear_acc = 100.0 * float((pred == yte).mean())
    te_idle = np.where(m_te & (y == 0))[0]
    idle_ok = 100.0 * float((rms[te_idle] <= thr).mean())
    overall = (wear_acc / 100 * len(wte) + idle_ok / 100 * len(te_idle)) / (len(wte) + len(te_idle)) * 100
    print(f"[test] wear3 {wear_acc:.2f}% | idle_gate {idle_ok:.2f}% | 4-class {overall:.2f}%", flush=True)

    ckpt = os.path.join(args.out, "vibnet_gate3_xyz.pt")
    torch.save({"state_dict": best_state,
                "config": {"c1": args.c1, "c2": args.c2, "c3": args.c3, "k1": args.k1, "s1": args.s1},
                "class_names": WEAR, "axes": "xyz", "gate_threshold": thr,
                "normalize": "joint rms over x/y/z, clip ±6"}, ckpt)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "VibNet-gate3-xyz", "split": args.split, "params": n_par,
                   "gate_threshold": thr, "wear_test_acc": wear_acc,
                   "idle_gate_acc": idle_ok, "overall4_acc": overall,
                   "best_val_wear_acc": best}, f, ensure_ascii=False, indent=2)
    print("[saved]", ckpt, flush=True)


if __name__ == "__main__":
    main()
