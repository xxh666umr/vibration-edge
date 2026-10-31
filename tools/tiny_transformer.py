# -*- coding: utf-8 -*-
"""Tiny Transformer 可行性实验（面向 STM32F103：512KB Flash / 64KB RAM / 72MHz）。

设计：3×1024 → patch embedding(conv k64 s32, d=24) → 32 个 token
     → 1 层 TransformerEncoder（2 头自注意力 + FFN 48）
     → 平均池化 → FC(3)
对比：同数据、同划分下的 1D CNN（本方案）。

用法：
    python tools/tiny_transformer.py --axes xyz --split random --epochs 60 --out models\tiny_transformer
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
from train_vib import VibNet, augment, split_block, split_random  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402


class TinyTransformer(nn.Module):
    def __init__(self, in_c=3, d=24, nhead=2, ff=48, layers=1, nc=3, in_len=1024):
        super().__init__()
        self.patch = nn.Conv1d(in_c, d, kernel_size=64, stride=32)
        self.n_tokens = (in_len - 64) // 32 + 1   # 31? 实际 31；用 31
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                         dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, nc)

    def forward(self, x):
        h = self.patch(x)                 # (B, d, T)
        h = h.transpose(1, 2)             # (B, T, d)
        h = h + self.pos[:, : h.shape[1], :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


def count_macs(model, shape):
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        macs[0] += out.numel() * m.in_channels * m.kernel_size[0] // m.groups

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    def attn_hook(m, inp, out):
        q = inp[0]
        if q.dim() == 3 and q.shape[0] != 1:
            L, B, E = q.shape
        elif q.dim() == 3:
            B, L, E = q.shape
        else:
            return
        macs[0] += 4 * L * B * E * E + 2 * L * L * B * E

    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(attn_hook))
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(shape, device=dev))
    for h in hooks:
        h.remove()
    return macs[0]


@torch.no_grad()
def predict(model, X, dev, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="xyz", choices=["x", "xyz"])
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--ff", type=int, default=48)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"models\tiny_transformer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    if args.axes == "xyz":
        X, y = load_xyz()
        in_c = 3
    else:
        X, y = load_xyz()
        X = X[:, :1, :]
        in_c = 1
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True
    wtr = np.where(m_tr & (y > 0))[0]
    wva = np.where(np.isin(np.arange(len(y)), va) & (y > 0))[0]
    wte = np.where(m_te & (y > 0))[0]
    Xtr = norm3(X[wtr], rms[wtr])
    Xva = norm3(X[wva], rms[wva])
    Xte = norm3(X[wte], rms[wte])
    ytr, yva, yte = y[wtr] - 1, y[wva] - 1, y[wte] - 1

    thr = float(rms[(y == 0) & np.isin(np.arange(len(y)), tr)].max()) * 3.0
    torch.manual_seed(args.seed)
    model = TinyTransformer(in_c, args.d, args.heads, args.ff, args.layers).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, (1, in_c, 1024))
    print(f"[data] {X.shape}, 训练/验证/测试 = {len(wtr)}/{len(wva)}/{len(wte)}")
    print(f"[model] TinyTransformer d={args.d} ff={args.ff} heads={args.heads} layers={args.layers} "
          f"params={n_par} MAC={macs} int8_KB={n_par/1024:.1f}")

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
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
            print(f"[ep {ep:3d}] loss={tot/len(Xtr):.4f} val={va_acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    pred = predict(model, Xte, dev)
    wear = 100.0 * float((pred == yte).mean())
    te_idle = np.where(m_te & (y == 0))[0]
    idle_ok = 100.0 * float((rms[te_idle] <= thr).mean())
    overall = (wear / 100 * len(wte) + idle_ok / 100 * len(te_idle)) / (len(wte) + len(te_idle)) * 100
    print(f"[test] wear3 {wear:.2f}% | idle_gate {idle_ok:.2f}% | 4-class {overall:.2f}%")

    # 同预算对比：1D CNN
    torch.manual_seed(args.seed)
    cnn = VibNet(16, 32, 48, 64, 32, nc=3, in_c=in_c).to(dev)
    n_cnn = sum(p.numel() for p in cnn.parameters())
    macs_cnn = count_macs(cnn, (1, in_c, 1024))
    ckpt = None
    for cand in ("models/gate3_xyz/vibnet_gate3_xyz.pt", "models/gate3/vibnet_gate3.pt"):
        if os.path.exists(cand):
            ckpt = torch.load(cand, map_location="cpu", weights_only=False)
            break
    cnn_acc = None
    if ckpt is not None and in_c == 3:
        cnn.load_state_dict(ckpt["state_dict"])
        cnn_acc = 100.0 * float((predict(cnn, Xte, dev) == yte).mean())
    print(f"[对比] 1D CNN params={n_cnn} MAC={macs_cnn} test_wear={cnn_acc if cnn_acc is None else round(cnn_acc,2)}%")

    # 部署预算估算（F103：512KB Flash / 64KB RAM / 72MHz）
    # 以实测 CNN 板端吞吐为基准：124.6k MAC / 31.4 ms（三轴）
    base_mac, base_ms = macs_cnn, 31.4
    est_ms = macs / base_mac * base_ms * 1.25   # 1.25 为 softmax/layernorm 等额外开销
    budget = {
        "model": "TinyTransformer", "axes": args.axes, "split": args.split,
        "d": args.d, "ff": args.ff, "heads": args.heads, "layers": args.layers,
        "params": n_par, "macs": macs, "int8_flash_kb": n_par / 1024,
        "est_board_ms": est_ms, "test_wear_acc": wear, "idle_gate": idle_ok,
        "overall4_acc": overall,
        "cnn_params": n_cnn, "cnn_macs": macs_cnn, "cnn_test_wear": cnn_acc,
        "baseline": {"mac": base_mac, "ms": base_ms, "note": "三轴 CNN 实测 31.4 ms"},
    }
    json.dump(budget, open(os.path.join(args.out, "metrics_tiny_transformer.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    torch.save({"state_dict": best_state, "config": vars(args)}, os.path.join(args.out, "tiny_transformer.pt"))
    print(f"[预算] params={n_par} -> int8 权重约 {n_par/1024:.1f} KB（Flash 512KB）；"
          f"MAC={macs} -> 板端估算 {est_ms:.1f} ms/窗（含 softmax/LN 开销）")


if __name__ == "__main__":
    main()
