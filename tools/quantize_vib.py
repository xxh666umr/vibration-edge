"""VibNet（工业振动 4 分类）int8 量化 + C 数组导出。

结构与 csrc/nn.h 对齐：
  conv1(valid) -> ReLU -> requant -> maxpool2
  conv2(valid) -> ReLU -> requant -> maxpool2
  conv3(k3,s1,pad1) -> ReLU -> requant -> GAP -> FC -> softmax

量化方案与调制识别一致：int8 对称逐张量、int32 累加、定点重量化（mul/2^24）、
round half away from zero；numpy 整数参考实现与 C 逐位对齐。

用法：
    python tools/quantize_vib.py --ckpt models/x_k64_80b/vibnet.pt --out models/x_k64_80b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vib import CLASSES, DATA_DIR, VibNet, evaluate, load_axis, split_random, split_block  # noqa: E402

FX_SHIFT = 24


def round_away(v):
    v = np.asarray(v, dtype=np.float64)
    return np.trunc(v + np.copysign(0.5, v))


def quantize_sym(x, scale, lo=-128, hi=127):
    return np.clip(round_away(np.asarray(x, np.float64) / float(scale)), lo, hi).astype(np.int64)


def fx_scale(req):
    return max(int(round(float(req) * (1 << FX_SHIFT))), 1), FX_SHIFT


def requantize_fx(acc, mul, shift, relu=True):
    p = np.asarray(acc, np.int64) * np.int64(mul) + (np.int64(1) << np.int64(shift - 1))
    r = p >> np.int64(shift)
    return np.clip(r, 0, 127) if relu else np.clip(r, -128, 127)


def conv1d_acc(xq, wq, bq, stride):
    """xq (N,Cin,Lin), wq (Cout,Cin,K), bq (Cout,) -> int32 (N,Cout,Lout)。"""
    wq = np.asarray(wq, np.int64)
    k = wq.shape[2]
    win = sliding_window_view(np.asarray(xq, np.int64), k, axis=2)[:, :, ::stride]
    return np.einsum("nctk,ock->not", win, wq, optimize=True) + np.asarray(bq, np.int64)[None, :, None]


def maxpool1d_int(xq, k, stride):
    win = sliding_window_view(np.asarray(xq, np.int64), k, axis=2)[:, :, ::stride]
    return win.max(axis=3)


def avgpool_global_int(x):
    """全局平均（k=L, stride=1），与 C 的 avgpool1d 一致：向零截断 + 四舍五入式加减。"""
    x = np.asarray(x, np.int64)
    n, c, l = x.shape
    s = x.sum(axis=2)
    k = l
    num = s + np.where(s >= 0, k // 2, -(k // 2))
    r = np.trunc(num / k).astype(np.int64)
    return np.clip(r, -128, 127)


def softmax_int8(logits, acc_scale):
    z = (np.asarray(logits, np.int64).astype(np.float32) * np.float32(acc_scale)).astype(np.float32)
    mx = z.max(axis=1, keepdims=True)
    e = np.exp((z - mx).astype(np.float64)).astype(np.float32)
    s = e.astype(np.float64).sum(axis=1, keepdims=True)
    return (e / s).astype(np.float32)


def quantize_vib(model, calib):
    import torch
    calib = np.asarray(calib, np.float32)
    n, c_in, l_in = calib.shape
    with torch.no_grad():
        w1 = model.conv1.weight.detach().cpu().numpy().astype(np.float64)
        b1 = model.conv1.bias.detach().cpu().numpy().astype(np.float64)
        w2 = model.conv2.weight.detach().cpu().numpy().astype(np.float64)
        b2 = model.conv2.bias.detach().cpu().numpy().astype(np.float64)
        w3 = model.conv3.weight.detach().cpu().numpy().astype(np.float64)
        b3 = model.conv3.bias.detach().cpu().numpy().astype(np.float64)
        wf = model.fc.weight.detach().cpu().numpy().astype(np.float64)
        bf = model.fc.bias.detach().cpu().numpy().astype(np.float64)
        k1, s1 = model.conv1.kernel_size[0], model.conv1.stride[0]
        k2, s2 = model.conv2.kernel_size[0], model.conv2.stride[0]
        k3 = model.conv3.kernel_size[0]
    c1, c2, c3 = w1.shape[0], w2.shape[0], w3.shape[0]
    l1 = (l_in - k1) // s1 + 1
    l1p = (l1 - 2) // 2 + 1
    l2 = (l1p - k2) // s2 + 1
    l2p = (l2 - 2) // 2 + 1
    l3in = l2p
    l3 = l3in  # k3=3, stride=1, padding=1

    s_in = max(float(np.abs(calib).max()) / 127.0, 1e-6)
    xq = quantize_sym(calib, s_in)

    s_w1 = max(float(np.abs(w1).max()) / 127.0, 1e-6)
    w1q = quantize_sym(w1, s_w1)
    acc1_scale = s_in * s_w1
    b1q = np.clip(round_away(b1 / acc1_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
    acc1 = conv1d_acc(xq, w1q, b1q, s1)
    out1_scale = max(float((np.maximum(acc1, 0) * acc1_scale).max()) / 127.0, 1e-6)
    mul1, sh1 = fx_scale(acc1_scale / out1_scale)
    h1 = requantize_fx(acc1, mul1, sh1, relu=True)
    p1 = maxpool1d_int(h1, 2, 2)

    s_w2 = max(float(np.abs(w2).max()) / 127.0, 1e-6)
    w2q = quantize_sym(w2, s_w2)
    acc2_scale = out1_scale * s_w2
    b2q = np.clip(round_away(b2 / acc2_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
    acc2 = conv1d_acc(p1, w2q, b2q, s2)
    out2_scale = max(float((np.maximum(acc2, 0) * acc2_scale).max()) / 127.0, 1e-6)
    mul2, sh2 = fx_scale(acc2_scale / out2_scale)
    h2 = requantize_fx(acc2, mul2, sh2, relu=True)
    p2 = maxpool1d_int(h2, 2, 2)

    # conv3：zero padding 1（在 numpy 参考里手工 pad）
    s_w3 = max(float(np.abs(w3).max()) / 127.0, 1e-6)
    w3q = quantize_sym(w3, s_w3)
    acc3_scale = out2_scale * s_w3
    b3q = np.clip(round_away(b3 / acc3_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
    pad = np.zeros((p2.shape[0], p2.shape[1], l3in + 2), np.int64)
    pad[:, :, 1:1 + l3in] = p2
    acc3 = conv1d_acc(pad, w3q, b3q, 1)
    out3_scale = max(float((np.maximum(acc3, 0) * acc3_scale).max()) / 127.0, 1e-6)
    mul3, sh3 = fx_scale(acc3_scale / out3_scale)
    h3 = requantize_fx(acc3, mul3, sh3, relu=True)

    gap = avgpool_global_int(h3)
    s_wf = max(float(np.abs(wf).max()) / 127.0, 1e-6)
    wfq = quantize_sym(wf, s_wf)
    fc_scale = out3_scale * s_wf
    bfq = np.clip(round_away(bf / fc_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
    logits = gap.astype(np.int64) @ wfq.T + bfq

    return {
        "input": {"c": int(c_in), "l": int(l_in), "scale": float(s_in)},
        "conv1": {"cin": int(c_in), "cout": int(c1), "k": int(k1), "s": int(s1),
                  "lin": int(l_in), "lout": int(l1), "w": w1q, "b": b1q,
                  "w_scale": float(s_w1), "acc_scale": float(acc1_scale),
                  "out_scale": float(out1_scale), "fx_mul": int(mul1), "fx_shift": int(sh1)},
        "pool1": {"l_in": int(l1), "l_out": int(l1p)},
        "conv2": {"cin": int(c1), "cout": int(c2), "k": int(k2), "s": int(s2),
                  "lin": int(l1p), "lout": int(l2), "w": w2q, "b": b2q,
                  "w_scale": float(s_w2), "acc_scale": float(acc2_scale),
                  "out_scale": float(out2_scale), "fx_mul": int(mul2), "fx_shift": int(sh2)},
        "pool2": {"l_in": int(l2), "l_out": int(l2p)},
        "conv3": {"cin": int(c2), "cout": int(c3), "k": int(k3), "s": 1,
                  "lin": int(l3in), "lout": int(l3), "w": w3q, "b": b3q,
                  "w_scale": float(s_w3), "acc_scale": float(acc3_scale),
                  "out_scale": float(out3_scale), "fx_mul": int(mul3), "fx_shift": int(sh3)},
        "gap": {"l_in": int(l3)},
        "fc": {"cin": int(c3), "cout": int(wf.shape[0]), "w": wfq, "b": bfq,
               "w_scale": float(s_wf), "acc_scale": float(fc_scale)},
        "param_count": int(w1q.size + b1q.size + w2q.size + b2q.size + w3q.size + b3q.size
                           + wfq.size + bfq.size),
        "weight_bytes": int(w1q.size + w2q.size + w3q.size + wfq.size),
        "bias_bytes": int(4 * (b1q.size + b2q.size + b3q.size + bfq.size)),
        "classes": CLASSES,
    }


def quantized_forward(q, features):
    x = np.asarray(features, np.float32)
    if x.ndim == 2:
        x = x.reshape(-1, q["input"]["c"], q["input"]["l"])
    xq = quantize_sym(x, q["input"]["scale"])
    c1, c2, c3, fc = q["conv1"], q["conv2"], q["conv3"], q["fc"]
    acc1 = conv1d_acc(xq, c1["w"], c1["b"], c1["s"])
    h1 = requantize_fx(acc1, c1["fx_mul"], c1["fx_shift"], True)
    p1 = maxpool1d_int(h1, 2, 2)
    acc2 = conv1d_acc(p1, c2["w"], c2["b"], c2["s"])
    h2 = requantize_fx(acc2, c2["fx_mul"], c2["fx_shift"], True)
    p2 = maxpool1d_int(h2, 2, 2)
    pad = np.zeros((p2.shape[0], p2.shape[1], c3["lin"] + 2), np.int64)
    pad[:, :, 1:1 + c3["lin"]] = p2
    acc3 = conv1d_acc(pad, c3["w"], c3["b"], 1)
    h3 = requantize_fx(acc3, c3["fx_mul"], c3["fx_shift"], True)
    gap = avgpool_global_int(h3)
    logits = gap.astype(np.int64) @ fc["w"].T + fc["b"]
    probs = softmax_int8(logits, fc["acc_scale"])
    return probs.argmax(1), probs, logits


# ---------------------------------------------------------------- C 导出
def _c_array(name, a, dtype, per=16):
    flat = np.asarray(a).reshape(-1)
    rows = [", ".join(str(int(v)) for v in flat[i:i + per]) for i in range(0, len(flat), per)]
    body = ",\n  ".join(rows)
    return f"static const {dtype} {name}[{len(flat)}] = {{\n  {body}\n}};"


def write_model(q, out_dir, name="vib_cnn_x"):
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "model_vib.h")
    c_path = os.path.join(out_dir, "model_vib.c")
    c1, c2, c3, fc = q["conv1"], q["conv2"], q["conv3"], q["fc"]
    h = f"""/* 自动生成：tools/quantize_vib.py（VibNet 工业振动 4 分类，int8） */
#ifndef MODEL_VIB_H
#define MODEL_VIB_H
#include <stdint.h>

#define MODEL_NAME "{name}"
#define MODEL_INPUT_C      {q['input']['c']}
#define MODEL_INPUT_LEN    {q['input']['l']}
#define MODEL_NUM_CLASSES  {len(q['classes'])}
#define MODEL_INPUT_SCALE  {q['input']['scale']:.9g}f
#define MODEL_PARAM_COUNT  {q['param_count']}
#define MODEL_CONV1_FX_MUL   {c1['fx_mul']}
#define MODEL_CONV1_FX_SHIFT {c1['fx_shift']}
#define MODEL_CONV2_FX_MUL   {c2['fx_mul']}
#define MODEL_CONV2_FX_SHIFT {c2['fx_shift']}
#define MODEL_CONV3_FX_MUL   {c3['fx_mul']}
#define MODEL_CONV3_FX_SHIFT {c3['fx_shift']}

#ifdef __cplusplus
extern "C" {{
#endif

int model_forward(const int8_t *input, float *probs);

#ifdef __cplusplus
}}
#endif
#endif /* MODEL_VIB_H */
"""
    l1, l1p = c1["lout"], q["pool1"]["l_out"]
    l2, l2p = c2["lout"], q["pool2"]["l_out"]
    l3 = c3["lout"]
    arrays = "\n\n".join([
        _c_array("W_CONV1", c1["w"], "int8_t"),
        _c_array("B_CONV1", c1["b"], "int32_t", per=8),
        _c_array("W_CONV2", c2["w"], "int8_t"),
        _c_array("B_CONV2", c2["b"], "int32_t", per=8),
        _c_array("W_CONV3", c3["w"], "int8_t"),
        _c_array("B_CONV3", c3["b"], "int32_t", per=8),
        _c_array("W_FC", fc["w"], "int8_t"),
        _c_array("B_FC", fc["b"], "int32_t", per=8),
    ])
    body = f"""/* 自动生成：tools/quantize_vib.py；输入 int8[1][1024]（x 轴振动），返回 argmax。 */
#include <string.h>
#include "model_vib.h"
#include "nn.h"

#define C1 {c1['cout']}
#define C2 {c2['cout']}
#define C3 {c3['cout']}
#define L1 {l1}
#define L1P {l1p}
#define L2 {l2}
#define L2P {l2p}
#define L3 {l3}

{arrays}

static int8_t g_h1[C1 * L1];
static int8_t g_p1[C1 * L1P];
static int8_t g_h2[C2 * L2];
static int8_t g_p2[C2 * L2P];
static int8_t g_pad[C2 * (L2P + 2)];
static int8_t g_h3[C3 * L3];
static int32_t g_acc1[C1 * L1];
static int32_t g_acc2[C2 * L2];
static int32_t g_acc3[C3 * L3];

int model_forward(const int8_t *input, float *probs)
{{
    int i, c, t;

    conv1d_accumulate(W_CONV1, B_CONV1, input, g_acc1, {q['input']['c']}, C1, MODEL_INPUT_LEN, {c1['k']}, {c1['s']});
    requantize_relu_fx(g_acc1, g_h1, C1 * L1, MODEL_CONV1_FX_MUL, MODEL_CONV1_FX_SHIFT);
    maxpool1d(g_h1, g_p1, C1, L1, 2, 2);

    conv1d_accumulate(W_CONV2, B_CONV2, g_p1, g_acc2, C1, C2, L1P, {c2['k']}, {c2['s']});
    requantize_relu_fx(g_acc2, g_h2, C2 * L2, MODEL_CONV2_FX_MUL, MODEL_CONV2_FX_SHIFT);
    maxpool1d(g_h2, g_p2, C2, L2, 2, 2);

    /* conv3: k=3, stride=1, padding=1 -> 先零填充到 L2P+2 */
    memset(g_pad, 0, sizeof(g_pad));
    for (c = 0; c < C2; c++) {{
        for (t = 0; t < L2P; t++) g_pad[c * (L2P + 2) + 1 + t] = g_p2[c * L2P + t];
    }}
    conv1d_accumulate(W_CONV3, B_CONV3, g_pad, g_acc3, C2, C3, L2P + 2, {c3['k']}, 1);
    requantize_relu_fx(g_acc3, g_h3, C3 * L3, MODEL_CONV3_FX_MUL, MODEL_CONV3_FX_SHIFT);

    /* 全局平均池化 -> C3 个 int8 */
    {{
        int8_t gap[C3];
        avgpool1d(g_h3, gap, C3, L3, L3, 1);
        {{
            int32_t acc[{len(q['classes'])}];
            matmul_accumulate(W_FC, B_FC, gap, acc, C3, {len(q['classes'])});
            return softmax_argmax(acc, {len(q['classes'])}, {fc['acc_scale']:.9g}f, probs);
        }}
    }}
}}
"""
    open(h_path, "w", encoding="utf-8").write(h)
    open(c_path, "w", encoding="utf-8").write(body)
    return h_path, c_path


def write_demo_sample(path, xq, true_idx, pred_idx, true_name, pred_name):
    arr = np.asarray(xq, np.int64).reshape(-1)
    rows = [", ".join(str(int(v)) for v in arr[i:i + 32]) for i in range(0, len(arr), 32)]
    body = ",\n  ".join(rows)
    txt = f"""/* 自动生成：一条真实测试样本（int8），供固件内置演示。 */
#ifndef DEMO_SAMPLE_VIB_H
#define DEMO_SAMPLE_VIB_H
#include <stdint.h>
#define DEMO_SAMPLE_TRUE {true_idx}
#define DEMO_SAMPLE_PRED {pred_idx}
#define DEMO_SAMPLE_TRUE_NAME "{true_name}"
#define DEMO_SAMPLE_PRED_NAME "{pred_name}"
static const int8_t DEMO_SAMPLE[{len(arr)}] = {{
  {body}
}};
#endif
"""
    open(path, "w", encoding="utf-8").write(txt)
    return path


def save_meta(q, path):
    meta = {
        "model": "vib_cnn_x",
        "classes": q["classes"],
        "input": q["input"],
        "arch": {
            "conv1": {k: q["conv1"][k] for k in ("cin", "cout", "k", "s", "lin", "lout")},
            "pool1": q["pool1"],
            "conv2": {k: q["conv2"][k] for k in ("cin", "cout", "k", "s", "lin", "lout")},
            "pool2": q["pool2"],
            "conv3": {k: q["conv3"][k] for k in ("cin", "cout", "k", "s", "lin", "lout")},
            "gap": q["gap"],
            "fc": {k: q["fc"][k] for k in ("cin", "cout")},
        },
        "scales": {
            "input": q["input"]["scale"],
            "conv1_out": q["conv1"]["out_scale"],
            "conv2_out": q["conv2"]["out_scale"],
            "conv3_out": q["conv3"]["out_scale"],
            "fc_acc": q["fc"]["acc_scale"],
        },
        "fx": {
            "conv1": [q["conv1"]["fx_mul"], q["conv1"]["fx_shift"]],
            "conv2": [q["conv2"]["fx_mul"], q["conv2"]["fx_shift"]],
            "conv3": [q["conv3"]["fx_mul"], q["conv3"]["fx_shift"]],
        },
        "param_count": q["param_count"],
        "weight_bytes": q["weight_bytes"],
        "bias_bytes": q["bias_bytes"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="random", choices=["random", "block"])
    ap.add_argument("--axes", default="x")
    ap.add_argument("--n-calib", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    in_c = len(args.axes)
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], in_c=in_c)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X, y = load_axis(args.axes)
    tr, va, te = split_random(y) if args.split == "random" else split_block(y)
    Xtr = X[tr] if in_c > 1 else X[tr][:, None, :]
    Xte = X[te] if in_c > 1 else X[te][:, None, :]

    calib = Xtr[:args.n_calib]
    q = quantize_vib(model, calib)
    pred_i8, probs, logits = quantized_forward(q, Xte)
    acc_i8 = float((pred_i8 == y[te]).mean() * 100)
    # 浮点对照
    pred_f = evaluate(model, torch.from_numpy(Xte).float(), y[te], torch.device("cpu"))["accuracy"]
    print(f"[float] {pred_f:.2f}%  [int8] {acc_i8:.2f}%  drop {pred_f - acc_i8:.2f} pt")
    print(f"[q] params={q['param_count']} weights={q['weight_bytes']}B bias={q['bias_bytes']}B")

    h, c = write_model(q, args.out)
    save_meta(q, os.path.join(args.out, "vib_meta.json"))
    di = int(np.where(pred_i8 == y[te])[0][0])
    demo_x = quantize_sym(Xte[di:di + 1], q["input"]["scale"])[0]
    write_demo_sample(os.path.join(args.out, "demo_sample_vib.h"), demo_x,
                      int(y[te][di]), int(pred_i8[di]),
                      CLASSES[int(y[te][di])], CLASSES[int(pred_i8[di])])
    np.savez_compressed(os.path.join(args.out, "parity_vib.npz"),
                        features=Xte[:200].astype(np.float32), labels=y[te][:200])
    with open(os.path.join(args.out, "metrics_int8.json"), "w", encoding="utf-8") as f:
        json.dump({"float_acc": pred_f, "int8_acc": acc_i8, "q": {"param_count": q["param_count"]}},
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {h}\n[saved] {c}")


if __name__ == "__main__":
    main()
