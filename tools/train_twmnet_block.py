"""TWM-Net 在"时间块划分"下的重跑（与随机口径对照用）。

时间块划分：每类前 70% 训练 / 中 20% 验证 / 后 10% 测试（保持时间顺序，避免相邻窗口泄漏）。
TWM-Net 结构来自上一阶段项目 src/model_twmnet.py（论文复现，21.6M 参数）。

用法：python tools/train_twmnet_block.py --epochs 50
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
sys.path.insert(0, r"C:\Users\lenovo\Desktop\工业语义通信Demo项目\src")

from model_twmnet import TWMNet, count_parameters  # noqa: E402
from train_vib import load_axis, split_block  # noqa: E402

CLASSES = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    acc = 100.0 * tp.sum() / cm.sum()
    return {"accuracy": float(acc), "confusion": cm.tolist()}


@torch.no_grad()
def evaluate(model, X, y, dev, batch=512):
    model.eval()
    cm = np.zeros((4, 4), np.int64)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(dev)
        p = model(xb).argmax(1).cpu().numpy()
        for t, q in zip(y[i:i + batch], p):
            cm[int(t), int(q)] += 1
    return metrics_from_cm(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"models\twmnet_block")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    X, y = load_axis("x")
    tr, va, te = split_block(y)
    print(f"[data] x-axis {X.shape}, block split train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    Xtr = torch.from_numpy(X[tr][:, None, :]).to(dev)
    ytr = torch.from_numpy(y[tr]).to(dev)
    Xva = torch.from_numpy(X[va][:, None, :]).to(dev)
    Xte = torch.from_numpy(X[te][:, None, :]).to(dev)

    torch.manual_seed(args.seed)
    model = TWMNet(num_classes=4, in_len=1024).to(dev)
    print(f"[model] TWM-Net params={count_parameters(model):,}", flush=True)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_va, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for i in range(0, len(tr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vm = evaluate(model, Xva, y[va], dev)
        if vm["accuracy"] > best_va:
            best_va = vm["accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}] loss={tot/len(tr):.4f} val_acc={vm['accuracy']:.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    res = {"model": "TWM-Net", "split": "block", "params": count_parameters(model),
           "epochs": args.epochs, "best_val_acc": best_va,
           "test": evaluate(model, Xte, y[te], dev)}
    print(f"[test] TWM-Net block-split acc={res['test']['accuracy']:.2f}%", flush=True)
    with open(os.path.join(args.out, "metrics_twmnet_block.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", os.path.join(args.out, "metrics_twmnet_block.json"), flush=True)


if __name__ == "__main__":
    main()
