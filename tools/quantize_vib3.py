# -*- coding: utf-8 -*-
"""三轴 gate3 模型的 int8 量化与 C 导出（输入 3×1024）。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_vib import (quantized_forward, quantize_sym, quantize_vib,  # noqa: E402
                          save_meta, write_demo_sample, write_model)
from train_vib import VibNet  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402
from train_vib import split_block, split_random  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--n-calib", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], nc=3)
    import torch.nn as nn
    model.conv1 = nn.Conv1d(3, cfg["c1"], cfg["k1"], cfg["s1"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X, y = load_xyz()
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True
    thr = float(ck["gate_threshold"])

    wtr = np.where(m_tr & (y > 0))[0]
    wte = np.where(m_te & (y > 0))[0]
    Xtr = norm3(X[wtr], rms[wtr])
    Xte = norm3(X[wte], rms[wte])
    yte = y[wte] - 1

    q = quantize_vib(model, Xtr[:args.n_calib])
    q["classes"] = ck["class_names"]
    pred, probs, _ = quantized_forward(q, Xte)
    wear = 100.0 * float((pred == yte).mean())
    with torch.no_grad():
        pf = model(torch.from_numpy(Xte).float()).argmax(1).numpy()
    wear_f = 100.0 * float((pf == yte).mean())
    te_idle = np.where(m_te & (y == 0))[0]
    idle_ok = 100.0 * float((rms[te_idle] <= thr).mean())
    overall = (wear / 100 * len(wte) + idle_ok / 100 * len(te_idle)) / (len(wte) + len(te_idle)) * 100
    print(f"[xyz wear] float {wear_f:.2f}% -> int8 {wear:.2f}% | idle_gate {idle_ok:.2f}% | 4-class {overall:.2f}%")
    print(f"[q] params={q['param_count']} weights={q['weight_bytes']}B bias={q['bias_bytes']}B")

    h, c = write_model(q, args.out, name="vib_cnn_gate3_xyz")
    save_meta(q, os.path.join(args.out, "vib_meta.json"))
    di = int(np.where(pred == yte)[0][0])
    demo_x = quantize_sym(Xte[di:di + 1], q["input"]["scale"])[0]
    write_demo_sample(os.path.join(args.out, "demo_sample_vib.h"), demo_x,
                      int(yte[di]), int(pred[di]), ck["class_names"][int(yte[di])], ck["class_names"][int(pred[di])])
    np.savez_compressed(os.path.join(args.out, "parity_vib.npz"),
                        features=Xte[:150].astype(np.float32), labels=yte[:150])
    with open(os.path.join(args.out, "metrics_int8.json"), "w", encoding="utf-8") as f:
        json.dump({"wear_float": wear_f, "wear_int8": wear, "idle_gate": idle_ok,
                   "overall4_int8": overall, "gate_threshold": thr,
                   "param_count": q["param_count"], "weight_bytes": q["weight_bytes"],
                   "bias_bytes": q["bias_bytes"], "axes": "xyz",
                   "classes": ck["class_names"]}, f, ensure_ascii=False, indent=2)
    print("[saved]", h, c)


if __name__ == "__main__":
    main()
