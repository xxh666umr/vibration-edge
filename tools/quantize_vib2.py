"""两级方案（能量门限 + 3 类归一化波形 CNN）的 int8 量化与 C 导出。

输入是**逐窗 RMS 归一化**后的波形（float 范围 ±6），int8 量化精度充足；
idling 由固件里的 RMS 门限判定（不进入 CNN）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_vib import (avgpool_global_int, conv1d_acc, fx_scale,  # noqa: E402
                          maxpool1d_int, quantize_sym, quantize_vib,
                          quantized_forward, requantize_fx, round_away,
                          save_meta, softmax_int8, write_demo_sample, write_model)
from train_vib import VibNet, split_block, split_random  # noqa: E402
from train_vib2 import WEAR_CLASSES, load_xyzr, normalize_wave  # noqa: E402

CLASSES4 = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=0.006)
    ap.add_argument("--split", default="random", choices=["random", "block"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    classes = ck.get("class_names", WEAR_CLASSES)
    nc = len(classes)
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], nc=nc)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X, y, rms = load_xyzr("x")
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True
    wear_tr = m_tr & (y > 0)
    wear_te = m_te & (y > 0)
    idle_te = m_te & (y == 0)

    wtr = np.where(wear_tr)[0]
    Xw_tr = normalize_wave(X[wtr], rms[wtr])
    wte = np.where(wear_te)[0]
    Xw_te = normalize_wave(X[wte], rms[wte])
    yw_te = y[wte] - 1

    q = quantize_vib(model, Xw_tr[:args.n_calib][:, None, :])
    q["classes"] = classes
    pred, probs, logits = quantized_forward(q, Xw_te[:, None, :])

    wear_acc_i8 = 100.0 * float((pred == yw_te).mean())
    # 浮点对照
    with torch.no_grad():
        pf = model(torch.from_numpy(Xw_te[:, None, :]).float()).argmax(1).numpy()
    wear_acc_f = 100.0 * float((pf == yw_te).mean())
    # idling 门限
    thr = float(args.thr)
    idle_ok = 100.0 * float((rms[np.where(idle_te)[0]] <= thr).mean())
    # 端到端 4 类：门限判空闲，> 门限才进 CNN（CNN 对全部非空闲样本预测磨损类别）
    Xw_all_te = normalize_wave(X[te], rms[te])
    pred_all, _, _ = quantized_forward(q, Xw_all_te[:, None, :])
    pred4 = np.where(rms[te] > thr, pred_all + 1, 0)
    overall = 100.0 * float((pred4 == y[te]).mean())
    idle_ok = 100.0 * float(((rms[te] <= thr) == (y[te] == 0)).mean())
    print(f"[wear] float {wear_acc_f:.2f}% -> int8 {wear_acc_i8:.2f}% (drop {wear_acc_f-wear_acc_i8:.2f} pt)")
    print(f"[idling gate] {idle_ok:.2f}% (thr={thr:.5f})")
    print(f"[4 类合计] int8 方案 {overall:.2f}%")
    print(f"[q] params={q['param_count']} weights={q['weight_bytes']}B bias={q['bias_bytes']}B")

    h, c = write_model(q, args.out, name="vib_cnn_gate3")
    save_meta(q, os.path.join(args.out, "vib_meta.json"))
    # 一条测试样本
    di = int(np.where(pred == yw_te)[0][0])
    demo_x = quantize_sym(Xw_te[di:di + 1], q["input"]["scale"])[0]
    write_demo_sample(os.path.join(args.out, "demo_sample_vib.h"), demo_x,
                      int(yw_te[di]), int(pred[di]),
                      classes[int(yw_te[di])], classes[int(pred[di])])
    np.savez_compressed(os.path.join(args.out, "parity_vib.npz"),
                        features=Xw_te[:200].astype(np.float32), labels=yw_te[:200])
    met = {"wear_float": wear_acc_f, "wear_int8": wear_acc_i8,
           "idle_gate": idle_ok, "overall4_int8": overall,
           "gate_threshold": thr,
           "param_count": q["param_count"], "weight_bytes": q["weight_bytes"],
           "bias_bytes": q["bias_bytes"], "classes": classes}
    with open(os.path.join(args.out, "metrics_int8.json"), "w", encoding="utf-8") as f:
        json.dump(met, f, ensure_ascii=False, indent=2)
    print(f"[saved] {h}\n[saved] {c}\n[saved] demo_sample_vib.h / vib_meta.json / parity_vib.npz")


if __name__ == "__main__":
    main()
