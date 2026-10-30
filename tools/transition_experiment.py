r"""过渡态方案实验：5 类 = 空闲(门限) + 初期/稳定/加速/过渡(CNN)。

设计（避免用测试集信息定义标签）：
1. 严格时间划分：每个类文件 前 80% 训练 / 80–90% 验证 / 90–100% 测试；
2. 阶段质心只用训练段里**前 60%** 的“核心窗口”计算（保证不含过渡段）；
3. 过渡标签规则：对 initial/steady 阶段的窗口，若其特征到"下一阶段质心"的
   标准化距离 < margin × 到"本阶段质心"的距离，则标为"过渡"（accelerating 没有下一阶段，不标）；
4. CNN 输出 4 类（initial/steady/accelerating/transition），空闲仍由 RMS 门限判。

用法：
    python tools/transition_experiment.py --axes x --margin 0.85
    python tools/transition_experiment.py --axes xyz --margin 0.85
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
from train_vib import VibNet, augment  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402

DATA_DIR = r"C:\Users\lenovo\Desktop\工业语义通信Demo项目\data\Toolweardata"
WEAR = ["initial_wear", "steady_wear", "accelerating_wear"]
LABELS4 = WEAR + ["transition"]


def load_x():
    xs, y = [], []
    for c in range(4):
        a = np.asarray(np.load(os.path.join(DATA_DIR, f"trainData_x_{c}.npy")), np.float32)
        xs.append(a)
        y.append(np.full(len(a), c, np.int64))
    return np.concatenate(xs), np.concatenate(y)


def features_1ch(X):
    rms = np.sqrt((X ** 2).mean(1))
    peak = np.abs(X).max(1)
    crest = peak / np.maximum(rms, 1e-9)
    mu = X.mean(1)
    sd = X.std(1) + 1e-9
    kurt = ((X - mu[:, None]) ** 4).mean(1) / sd ** 4
    zcr = (np.abs(np.diff(np.signbit(X).astype(np.int8), axis=1)) > 0).mean(1)
    Xf = np.abs(np.fft.rfft(X, axis=1))
    f = np.fft.rfftfreq(X.shape[1])
    total = Xf.sum(1) + 1e-12
    cent = (Xf * f).sum(1) / total
    bands = []
    for a, b in zip([0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.5]):
        m = (f >= a) & (f < b)
        bands.append(Xf[:, m].sum(1) / total)
    return np.stack([np.log(rms + 1e-9), cent, kurt, crest, zcr] + bands, 1).astype(np.float32)


def features_3ch(X):
    out = []
    for ax in range(3):
        f = features_1ch(X[:, ax, :])
        out.append(f)
    rms = np.sqrt((X ** 2).mean(2))  # (N,3)
    out.append(np.log(rms / (rms.sum(1, keepdims=True) + 1e-9) + 1e-9))  # 轴向能量占比
    return np.concatenate(out, 1).astype(np.float32)


def stage_split(y, tr_frac=0.8, va_frac=0.1):
    tr, va, te = [], [], []
    for c in range(4):
        idx = np.where(y == c)[0]
        n = len(idx)
        a, b = int(tr_frac * n), int((tr_frac + va_frac) * n)
        tr.extend(idx[:a]); va.extend(idx[a:b]); te.extend(idx[b:])
    return np.asarray(tr), np.asarray(va), np.asarray(te)


def make_transition_labels(y, feats, tr, margin):
    """用训练段前 60% 核心窗口算质心，返回 4 类标签与统计。"""
    # 标准化特征（用训练段前 60% 的均值方差）
    core_mask = np.zeros(len(y), bool)
    for c in (1, 2, 3):
        idx = np.where(y == c)[0]
        n = len(idx)
        core_mask[idx[:int(0.6 * n)]] = True
    mu, sd = feats[core_mask].mean(0), feats[core_mask].std(0) + 1e-9
    Z = (feats - mu) / sd

    cent = {}
    for c in (1, 2, 3):
        sel = np.where(core_mask & (y == c))[0]
        cent[c] = Z[sel].mean(0)

    lab = np.full(len(y), -1, np.int64)   # -1 = idle（交给门限）
    for c in (1, 2, 3):
        idx = np.where(y == c)[0]
        d_own = ((Z[idx] - cent[c]) ** 2).sum(1)
        if c < 3:
            d_next = ((Z[idx] - cent[c + 1]) ** 2).sum(1)
            is_tr = d_next < margin * d_own
        else:
            is_tr = np.zeros(len(idx), bool)
        lab[idx] = c - 1
        lab[idx[is_tr]] = 3              # transition
    stats = {c: int(((y == c) & (lab == 3)).sum()) for c in (1, 2, 3)}
    return lab, stats


@torch.no_grad()
def predict(model, X, dev, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="x", choices=["x", "xyz"])
    ap.add_argument("--margin", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"models\trans")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    if args.axes == "x":
        X, y = load_x()
        X = X[:, None, :]
        feats = features_1ch(X[:, 0, :])
    else:
        X, y = load_xyz()
        feats = features_3ch(X)
    in_c = X.shape[1]
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = stage_split(y)
    lab, stats = make_transition_labels(y, feats, tr, args.margin)
    print(f"[data] {X.shape}, 过渡标签数（initial/steady/acc）: {stats}", flush=True)
    for name, idxs in (("train", tr), ("val", va), ("test", te)):
        cnt = np.bincount(lab[idxs][lab[idxs] >= 0], minlength=4)
        print(f"[{name}] n={len(idxs)} 4类分布={cnt.tolist()} (idle={int((lab[idxs] < 0).sum())})", flush=True)

    thr = float(rms[(y == 0) & np.isin(np.arange(len(y)), tr)].max()) * 3.0
    wtr = tr[y[tr] > 0]; wva = va[y[va] > 0]; wte = te[y[te] > 0]

    def prep(idx):
        if in_c == 1:
            return norm3(X[idx][:, 0:1, :], rms[idx])
        return norm3(X[idx], rms[idx])

    Xtr = torch.from_numpy(prep(wtr)).to(dev)
    ytr = torch.from_numpy(lab[wtr]).to(dev)
    Xva = prep(wva)
    yva = lab[wva]
    Xte = prep(wte)
    yte = lab[wte]

    torch.manual_seed(args.seed)
    model = VibNet(16, 32, 48, 64, 32, nc=4).to(dev)
    if in_c == 3:
        model.conv1 = nn.Conv1d(3, 16, 64, 32).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor(
        [1.0, 1.0, 1.0, 1.2], device=dev))  # 过渡类稍加权
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = crit(model(augment(Xtr[idx], gen)), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        pv = predict(model, Xva, dev)
        acc = 100.0 * float((pv == yva).mean())
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}] loss={tot/len(Xtr):.4f} val4_acc={acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    pred = predict(model, Xte, dev)
    cm = np.zeros((4, 4), np.int64)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    wear4 = 100.0 * float(np.trace(cm) / cm.sum())
    # 粗粒度 3 类准确率（把 transition 预测映射到"本阶段/下一阶段"最好情况：不计）
    idle_ok = 100.0 * float((rms[te[y[te] == 0]] <= thr).mean()) if (y[te] == 0).any() else 100.0
    n_idle = int((y[te] == 0).sum())
    overall5 = (wear4 / 100 * len(wte) + idle_ok / 100 * n_idle) / (len(wte) + n_idle) * 100
    rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0) * 100
    print(f"\n[test] 4类(wear+transition) acc={wear4:.2f}% | idle_gate={idle_ok:.2f}% | 5类总体={overall5:.2f}%")
    print(f"[per-class recall] " + ", ".join(f"{LABELS4[i]}:{rec[i]:.1f}%" for i in range(4)))
    print("confusion (rows true):")
    print(cm)
    torch.save({"state_dict": best_state, "config": {"c1": 16, "c2": 32, "c3": 48, "k1": 64, "s1": 32},
                "class_names": LABELS4, "axes": args.axes, "margin": args.margin,
                "gate_threshold": thr}, os.path.join(args.out, f"trans_{args.axes}.pt"))
    json.dump({"axes": args.axes, "margin": args.margin, "params": n_par,
               "transition_counts": stats, "wear4_acc": wear4, "idle_gate": idle_ok,
               "overall5_acc": overall5, "per_class_recall": rec.tolist(),
               "confusion": cm.tolist(), "gate_threshold": thr},
              open(os.path.join(args.out, f"metrics_{args.axes}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
