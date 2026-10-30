r"""标签重构实验：用信号特征重新定义磨损阶段（最近质心 + 边界带→过渡）。

背景：原始 4 类标签按“阶段”粗分，每个阶段文件末段已进入下一阶段幅值区间，
导致严格时间划分时末段标签与信号矛盾（3 类模型在末段只有 ~15~38%）。
本实验不依赖外部 VB 数据，用**训练段核心窗口**估计各阶段特征质心，再按
“最近质心”重标全部窗口；若最近与次近两个阶段距离接近（比值 < band），标为“过渡”。
然后严格按时间划分（前 80% 训练 / 90–100% 测试）重新训练与评估。

用法：
    python tools/relabel_experiment.py --axes x --band 1.15
    python tools/relabel_experiment.py --axes xyz --band 1.15
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
from transition_experiment import features_1ch, features_3ch, load_x, stage_split  # noqa: E402

WEAR = ["initial_wear", "steady_wear", "accelerating_wear"]
TRANS = "transition"


@torch.no_grad()
def predict(model, X, dev, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def relabel(y, feats, band, core_frac=0.6, use_transition=True):
    """返回 (标签, 过渡数量, 质量指标)。标签：0/1/2 磨损阶段，3 过渡，-1 空闲。"""
    core = np.zeros(len(y), bool)
    for c in (1, 2, 3):
        idx = np.where(y == c)[0]
        core[idx[:int(core_frac * len(idx))]] = True
    mu, sd = feats[core].mean(0), feats[core].std(0) + 1e-9
    Z = (feats - mu) / sd
    cent = np.stack([Z[core & (y == c)].mean(0) for c in (1, 2, 3)])
    d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(2)  # (N,3)
    order = np.argsort(d, 1)
    near, second = order[:, 0], order[:, 1]
    lab = np.full(len(y), -1, np.int64)
    wear = y > 0
    lab[wear] = near[wear]
    n_trans = 0
    if use_transition:
        ratio = np.sqrt(d[wear, second[wear]]) / np.sqrt(d[wear, near[wear]] + 1e-12)
        is_tr = ratio < band
        w_idx = np.where(wear)[0]
        lab[w_idx[is_tr]] = 3
        n_trans = int(is_tr.sum())
    # 与原始标签一致率（仅统计非过渡的磨损窗口）
    orig = (near[wear] == (y[wear] - 1)).mean() * 100
    return lab, n_trans, float(orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="x", choices=["x", "xyz"])
    ap.add_argument("--band", type=float, default=1.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-transition", action="store_true")
    ap.add_argument("--out", default=r"models\relabel")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    if args.axes == "x":
        X, y = load_x()
        X = X[:, None, :]
        feats = features_1ch(X[:, 0, :])
        in_c = 1
    else:
        X, y = load_xyz()
        feats = features_3ch(X)
        in_c = 3
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = stage_split(y)
    lab, n_trans, agree = relabel(y, feats, args.band, use_transition=not args.no_transition)
    n_cls = 3 if args.no_transition else 4
    names = WEAR if args.no_transition else WEAR + [TRANS]
    print(f"[labels] 最近质心与原始标签一致率（磨损窗口）={agree:.2f}%，过渡窗口数={n_trans}", flush=True)
    for nm, idxs in (("train", tr), ("val", va), ("test", te)):
        cnt = np.bincount(lab[idxs][lab[idxs] >= 0], minlength=n_cls)
        print(f"[{nm}] n={len(idxs)} 分布={cnt.tolist()} (idle={int((lab[idxs] < 0).sum())})", flush=True)

    thr = float(rms[(y == 0) & np.isin(np.arange(len(y)), tr)].max()) * 3.0
    wtr, wva, wte = tr[y[tr] > 0], va[y[va] > 0], te[y[te] > 0]

    def prep(idx):
        return norm3(X[idx], rms[idx]) if in_c == 3 else norm3(X[idx][:, 0:1, :], rms[idx])

    Xtr = torch.from_numpy(prep(wtr)).to(dev)
    ytr = torch.from_numpy(lab[wtr]).to(dev)
    Xva, yva = prep(wva), lab[wva]
    Xte, yte = prep(wte), lab[wte]

    torch.manual_seed(args.seed)
    model = VibNet(16, 32, 48, 64, 32, nc=n_cls).to(dev)
    if in_c == 3:
        model.conv1 = nn.Conv1d(3, 16, 64, 32).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
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
            print(f"[ep {ep:3d}] loss={tot/len(Xtr):.4f} val={acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    pred = predict(model, Xte, dev)
    cm = np.zeros((n_cls, n_cls), np.int64)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    acc_corr = 100.0 * float(np.trace(cm) / cm.sum())
    # 用原始标签评估（过渡预测视为错）
    orig_map = {0: 0, 1: 1, 2: 2}
    ok_orig = 0
    for t, p, orig_c in zip(yte, pred, y[wte] - 1):
        ok_orig += int((p == orig_c) and (p in orig_map))
    acc_orig = 100.0 * ok_orig / len(yte)
    n_idle = int((y[te] == 0).sum())
    idle_ok = 100.0 * float((rms[np.where((y == 0) & np.isin(np.arange(len(y)), te))[0]] <= thr).mean())
    overall5 = (acc_corr / 100 * len(wte) + idle_ok / 100 * n_idle) / (len(wte) + n_idle) * 100
    print(f"\n[test] 信号重标口径 4类 acc={acc_corr:.2f}% | idle_gate={idle_ok:.2f}% | 5类总体={overall5:.2f}%")
    print(f"[test] 原始标签口径 acc={acc_orig:.2f}%（过渡预测计错）")
    print("confusion (rows true):")
    print(cm)
    torch.save({"state_dict": best_state,
                "config": {"c1": 16, "c2": 32, "c3": 48, "k1": 64, "s1": 32},
                "class_names": names, "axes": args.axes, "band": args.band,
                "gate_threshold": thr}, os.path.join(args.out, f"relabel_{args.axes}.pt"))
    json.dump({"axes": args.axes, "band": args.band, "params": n_par,
               "agree_original_pct": agree, "transition_windows": n_trans,
               "test_signal_relabel_acc": acc_corr, "test_original_label_acc": acc_orig,
               "idle_gate": idle_ok, "overall5_acc": overall5,
               "confusion": cm.tolist(), "gate_threshold": thr,
               "class_names": names},
              open(os.path.join(args.out, f"metrics_{args.axes}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
