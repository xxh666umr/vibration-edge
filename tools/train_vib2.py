r"""工业振动 4 分类的两级方案（MCU 友好，int8 友好）：

一级：能量门限。按窗口 RMS 判 idling（空闲）——RMS 明显低于磨损态，阈值固定，几乎零成本；
二级：3 类磨损分类 CNN。输入是**逐窗 RMS 归一化**的波形（x/rms），
      消除 27 倍幅度差导致的 int8 量化精度问题，专注波形/频谱形状。

输出：4 类准确率 = (idling 门限正确 + 3 类 CNN 正确) / 总数。

用法：
    python tools/train_vib2.py --epochs 80 --out models\gate3
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
from train_vib import (CLASSES, DATA_DIR, VibNet, augment, metrics_from_cm,  # noqa: E402
                       split_random, split_block)

WEAR_CLASSES = ["initial_wear", "steady_wear", "accelerating_wear"]
CLIP = 6.0  # 归一化波形裁剪（int8 scale 由校准集最大绝对值得出）


def load_xyzr(axis="x"):
    """返回 X (N,1024) float32, y (N,), rms (N,) float32。"""
    xs, ys, rs = [], [], []
    for c in range(4):
        a = np.asarray(np.load(os.path.join(DATA_DIR, f"trainData_{axis}_{c}.npy")), np.float32)
        xs.append(a)
        ys.append(np.full(len(a), c, np.int64))
        rs.append(np.sqrt((a ** 2).mean(axis=1)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)


def normalize_wave(X, rms, clip=CLIP):
    xn = X / np.maximum(rms[:, None], 1e-9)
    return np.clip(xn, -clip, clip).astype(np.float32)


@torch.no_grad()
def predict_probs(model, X, dev, batch=1024):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(dev)
        if xb.dim() == 2:
            xb = xb.unsqueeze(1)
        outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs)


def acc4_from_parts(pred_wear, y_wear, pred_idle, y_idle):
    """合并 4 类准确率：idling 用门限结果，其余用 CNN 结果。"""
    correct = int((pred_wear == y_wear).sum()) + int((pred_idle == y_idle).sum())
    total = len(y_wear) + len(y_idle)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="x")
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--c3", type=int, default=48)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--s1", type=int, default=32)
    ap.add_argument("--idle-thr", type=float, default=None, help="默认取 idling 训练 RMS 最大值×5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    X, y, rms = load_xyzr(args.axis)
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_va = np.zeros(len(y), bool); m_va[va] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True

    # 能量门限：用训练集 idling RMS 的上界估计（留 5 倍余量）
    if args.idle_thr is None:
        thr = float(rms[(y == 0) & m_tr].max() * 5.0)
    else:
        thr = float(args.idle_thr)
    print(f"[gate] idling) rms 训练集 max={rms[(y == 0) & m_tr].max():.5f}, "
          f"threshold={thr:.5f}", flush=True)

    # 各 split 中 idling 用门限评估，其余三类训练 CNN
    def parts(mask):
        idle = mask & (y == 0)
        wear = mask & (y > 0)
        return np.where(idle)[0], np.where(wear)[0]

    tr_idle, tr_wear = parts(m_tr)
    va_idle, va_wear = parts(m_va)
    te_idle, te_wear = parts(m_te)
    print(f"[split={args.split}] wear train {len(tr_wear)} val {len(va_wear)} "
          f"test {len(te_wear)} | idling test {len(te_idle)}", flush=True)

    # 训练数据：3 类磨损，标签映射 1->0,2->1,3->2；输入逐窗归一化
    Xw_tr = normalize_wave(X[tr_wear], rms[tr_wear])
    yw_tr = y[tr_wear] - 1
    Xw_va = normalize_wave(X[va_wear], rms[va_wear])
    yw_va = y[va_wear] - 1
    Xw_te = normalize_wave(X[te_wear], rms[te_wear])
    yw_te = y[te_wear] - 1

    Xtr = torch.from_numpy(Xw_tr[:, None, :]).to(dev)
    ytr = torch.from_numpy(yw_tr).to(dev)
    Xva = torch.from_numpy(Xw_va[:, None, :]).to(dev)

    torch.manual_seed(args.seed)
    model = VibNet(args.c1, args.c2, args.c3, args.k1, args.s1, nc=3).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    best_va, best_state, bad = 0.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot, corr = 0.0, 0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(Xtr[idx], gen)
            yb = ytr[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            corr += (model(xb).argmax(1) == yb).sum().item()
        sched.step()
        with torch.no_grad():
            pv = predict_probs(model, Xw_va, dev).argmax(1)
        va = 100.0 * float((pv == yw_va).mean())
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}] loss={tot/len(Xtr):.4f} train_acc={100*corr/len(Xtr):.2f}% "
                  f"wear_val_acc={va:.2f}% ({time.time()-t0:.0f}s)", flush=True)
        if bad >= 20 and ep > 30:
            print(f"[early stop] ep {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    # 浮点评估
    pw = predict_probs(model, Xw_te, dev).argmax(1)
    wear_acc = 100.0 * float((pw == yw_te).mean())
    # idling 门限评估
    idle_pred = (rms[te_idle] > thr).astype(np.int64)  # 应为 0（非空闲）才对
    idle_acc = 100.0 * float((idle_pred == 0).mean())
    acc4 = acc4_from_parts(pw, yw_te, idle_pred, np.zeros(len(te_idle), np.int64))
    cm_wear = np.zeros((3, 3), np.int64)
    for t, p in zip(yw_te, pw):
        cm_wear[t, p] += 1
    met = metrics_from_cm(np.pad(cm_wear, ((0, 1), (0, 1))))
    print(f"[test] wear(3类) {wear_acc:.2f}% | idling 门限 {idle_acc:.2f}% "
          f"(n={len(te_idle)}) | 4 类合计 {acc4:.2f}%", flush=True)
    print(f"[test] per-class F1(wear): "
          f"{ {WEAR_CLASSES[i]: round(met['per_class_f1'][CLASSES[i+1]], 1) for i in range(3)} }",
          flush=True)

    res = {
        "model": "VibNet-gate3", "axis": args.axis, "split": args.split,
        "params": n_par, "config": {k: getattr(args, k) for k in ("c1", "c2", "c3", "k1", "s1")},
        "gate_threshold": thr, "normalize": "per-window rms, clip ±6",
        "wear_test_acc": wear_acc, "idle_gate_acc": idle_acc, "overall4_acc": acc4,
        "best_wear_val_acc": best_va,
        "wear_confusion": cm_wear.tolist(),
        "idle_test_rms_max": float(rms[te_idle].max()) if len(te_idle) else None,
        "n_test": {"wear": int(len(te_wear)), "idle": int(len(te_idle))},
        "time_seconds": time.time() - t0,
    }
    ckpt = os.path.join(args.out, "vibnet_gate3.pt")
    torch.save({"state_dict": best_state, "config": res["config"],
                "class_names": WEAR_CLASSES, "gate_threshold": thr,
                "normalize": res["normalize"]}, ckpt)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[saved] {ckpt}", flush=True)


if __name__ == "__main__":
    main()
