r"""把“信号重标”三轴模型导出为 C（int8），用于板端跑修正标签版本。

与 quantize_vib3 的区别：标签采用 relabel 规则（最近质心阶段），
训练/测试划分为前 90% / 最后 10%（与 relabel_experiment 一致），
并把校准特征保存下来供逐样本对齐测试复用，保证量化参数完全一致。

用法：
    python tools/export_relabel_c.py --ckpt models\relabel_xyz_t90\relabel_xyz.pt --out models\relabel_xyz_t90
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_vib import (quantize_sym, quantize_vib, quantized_forward,  # noqa: E402
                          save_meta, write_demo_sample, write_model)
from relabel_experiment import relabel  # noqa: E402
from train_vib import VibNet  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402
from transition_experiment import features_3ch, stage_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"models\relabel_xyz_t90\relabel_xyz.pt")
    ap.add_argument("--out", default=r"models\relabel_xyz_t90")
    ap.add_argument("--n-calib", type=int, default=1500)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], nc=3)
    model.conv1 = nn.Conv1d(3, cfg["c1"], cfg["k1"], cfg["s1"])
    model.load_state_dict(ck["state_dict"]); model.eval()

    X, y = load_xyz()
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = stage_split(y, 0.9, 0.05)
    feats = features_3ch(X)
    lab, _, agree = relabel(y, feats, 1.15, use_transition=False)
    thr = float(ck["gate_threshold"])

    wtr = tr[y[tr] > 0]
    Xtr = norm3(X[wtr], rms[wtr])
    wte = te[y[te] > 0]
    Xte = norm3(X[wte], rms[wte])
    yte = lab[wte]

    q = quantize_vib(model, Xtr[:args.n_calib])
    q["classes"] = ck["class_names"]
    pred, probs, _ = quantized_forward(q, Xte)
    wear = 100.0 * float((pred == yte).mean())
    orig = 100.0 * float((pred == (y[wte] - 1)).mean())
    with torch.no_grad():
        pf = model(torch.from_numpy(Xte).float()).argmax(1).numpy()
    wear_f = 100.0 * float((pf == yte).mean())
    te_idle = np.where((y == 0) & np.isin(np.arange(len(y)), te))[0]
    idle_ok = 100.0 * float((rms[te_idle] <= thr).mean())
    overall = (wear / 100 * len(wte) + idle_ok / 100 * len(te_idle)) / (len(wte) + len(te_idle)) * 100
    print(f"[relabel wear] float {wear_f:.2f}% -> int8 {wear:.2f}% | idle_gate {idle_ok:.2f}% | 5类总体 {overall:.2f}%")
    print(f"[relabel 对原始标签] {orig:.2f}%")
    print(f"[q] params={q['param_count']} weights={q['weight_bytes']}B bias={q['bias_bytes']}B")

    h, c = write_model(q, out, name="vib_cnn_gate3_relabel")
    save_meta(q, os.path.join(out, "vib_meta.json"))
    di = int(np.where(pred == yte)[0][0])
    demo_x = quantize_sym(Xte[di:di + 1], q["input"]["scale"])[0]
    write_demo_sample(os.path.join(out, "demo_sample_vib.h"), demo_x,
                      int(yte[di]), int(pred[di]),
                      ck["class_names"][int(yte[di])], ck["class_names"][int(pred[di])])
    np.savez_compressed(os.path.join(out, "parity_vib.npz"),
                        features=Xte[:150].astype(np.float32), labels=yte[:150])
    np.save(os.path.join(out, "calib_features.npy"), Xtr[:args.n_calib].astype(np.float32))
    with open(os.path.join(out, "metrics_int8.json"), "w", encoding="utf-8") as f:
        json.dump({"wear_float": wear_f, "wear_int8": wear, "idle_gate": idle_ok,
                   "overall5_int8": overall, "original_label_acc": orig,
                   "gate_threshold": thr, "param_count": q["param_count"],
                   "weight_bytes": q["weight_bytes"], "bias_bytes": q["bias_bytes"],
                   "axes": "xyz", "classes": ck["class_names"],
                   "split": "前90%训练/后10%测试（信号重标标签）",
                   "relabel_agree": agree}, f, ensure_ascii=False, indent=2)
    print("[saved]", h, c)


if __name__ == "__main__":
    main()
