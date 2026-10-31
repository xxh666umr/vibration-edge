# -*- coding: utf-8 -*-
"""把 tiny Transformer（d=16）导出为“int8 权重 + float 激活”的 C 数组，并做 numpy 量化仿真。

设计：
- 权重按张量对称量化成 int8（Flash 约 5.9 KB）；
- 激活保持 float32（RAM 约 20 KB，F103 64 KB 够）；
- 矩阵乘：输入 float 量化成 int8（固定 scale）→ int8×int8 → int32 → 反量化回 float。
- LayerNorm / softmax / GELU 用 float（数量少，M3 软浮点可接受）。

用法：
    python tools/export_transformer_c.py --ckpt models\tiny_transformer_d16\tiny_transformer.pt --out models\tiny_transformer_d16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vib import split_random, split_block  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402
from tiny_transformer import TinyTransformer  # noqa: E402

PATCH_X_S = np.float32(6.0 / 127.0)
LN_X_S = np.float32(4.0 / 127.0)
ATTN_X_S = np.float32(1.0 / 127.0)
FFN_X_S = np.float32(4.0 / 127.0)
HEAD_X_S = np.float32(4.0 / 127.0)


def quant_w(w):
    w = np.asarray(w, np.float32)
    s = float(np.abs(w).max() / 127.0 + 1e-12)
    q = np.clip(np.round(w / s), -128, 127).astype(np.int8)
    return q, np.float32(s)


def qmatmul(w, s_w, x, s_x, bias):
    """x: (..., in) float → quantize → int8 matmul → dequant + bias."""
    xi = np.clip(np.round(x / s_x), -127, 127).astype(np.int32)
    acc = xi @ w.astype(np.int32).T
    return acc.astype(np.float32) * (s_w * s_x) + np.asarray(bias, np.float32)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class QuantTransformer:
    def __init__(self, ckpt, d=16, heads=2, ff=32, stride=32, tokens=31):
        sd = ckpt["state_dict"]
        self.d, self.heads, self.ff = d, heads, ff
        self.stride, self.T = stride, tokens
        self.hd = d // heads
        self.pos = sd["pos"].numpy().astype(np.float32)[0]          # (T,d)
        self.patch_w, self.patch_s = quant_w(sd["patch.weight"].numpy())
        self.patch_b = sd["patch.bias"].numpy().astype(np.float32)
        self.qkv_w, self.qkv_s = quant_w(sd["encoder.layers.0.self_attn.in_proj_weight"].numpy())
        self.qkv_b = sd["encoder.layers.0.self_attn.in_proj_bias"].numpy().astype(np.float32)
        self.out_w, self.out_s = quant_w(sd["encoder.layers.0.self_attn.out_proj.weight"].numpy())
        self.out_b = sd["encoder.layers.0.self_attn.out_proj.bias"].numpy().astype(np.float32)
        self.fc1_w, self.fc1_s = quant_w(sd["encoder.layers.0.linear1.weight"].numpy())
        self.fc1_b = sd["encoder.layers.0.linear1.bias"].numpy().astype(np.float32)
        self.fc2_w, self.fc2_s = quant_w(sd["encoder.layers.0.linear2.weight"].numpy())
        self.fc2_b = sd["encoder.layers.0.linear2.bias"].numpy().astype(np.float32)
        self.ln1_g = sd["encoder.layers.0.norm1.weight"].numpy().astype(np.float32)
        self.ln1_b = sd["encoder.layers.0.norm1.bias"].numpy().astype(np.float32)
        self.ln2_g = sd["encoder.layers.0.norm2.weight"].numpy().astype(np.float32)
        self.ln2_b = sd["encoder.layers.0.norm2.bias"].numpy().astype(np.float32)
        self.lnf_g = sd["norm.weight"].numpy().astype(np.float32)
        self.lnf_b = sd["norm.bias"].numpy().astype(np.float32)
        self.head_w, self.head_s = quant_w(sd["head.weight"].numpy())
        self.head_b = sd["head.bias"].numpy().astype(np.float32)
        self.scales = {"patch": float(PATCH_X_S), "attn": float(PATCH_X_S), "ctx": float(ATTN_X_S),
                       "ffn": float(LN_X_S), "gelu": float(FFN_X_S), "head": float(HEAD_X_S)}


    def _fmm(self, w, s, x, b):
        return x @ (w.astype(np.float32).T * s) + b

    def _attn_float(self, h):
        """h: (B,T,d)；返回 attention 输出（float）。"""
        B, T, _ = h.shape
        qkv = self._fmm(self.qkv_w, self.qkv_s, h, self.qkv_b)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        attn = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.hd), axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
        return self._fmm(self.out_w, self.out_s, ctx, self.out_b), ctx

    def calibrate(self, X):
        """用 float 权重统计各阶段激活 maxabs（post-LN 结构）。"""
        T = self.T
        B = X.shape[0]
        cols = np.empty((B, T, 192), np.float32)
        for t in range(T):
            cols[:, t, :] = X[:, :, t * self.stride:t * self.stride + 64].reshape(B, -1)
        stats = {"patch": float(np.abs(cols).max())}
        h = self._fmm(self.patch_w.reshape(self.d, -1), self.patch_s, cols, self.patch_b) + self.pos[:T][None]
        stats["attn"] = float(np.abs(h).max())                 # attention 输入
        attn_out, ctx = self._attn_float(h)
        stats["ctx"] = float(np.abs(ctx).max())                # 注意力上下文
        h = layernorm(h + attn_out, self.ln1_g, self.ln1_b)
        stats["ffn"] = float(np.abs(h).max())                  # FFN 输入（LN1 输出）
        f = gelu_tanh(self._fmm(self.fc1_w, self.fc1_s, h, self.fc1_b))
        stats["gelu"] = float(np.abs(f).max())                 # GELU 输出
        h = layernorm(h + self._fmm(self.fc2_w, self.fc2_s, f, self.fc2_b), self.lnf_g, self.lnf_b)
        stats["head"] = float(np.abs(h.mean(axis=1)).max())    # 池化后
        return stats

    def forward(self, x):
        """x: (B,3,1024) float normalized；post-LN 与 PyTorch 一致。"""
        B = x.shape[0]
        T = self.T
        cols = np.empty((B, T, 3 * 64), np.float32)
        for t in range(T):
            cols[:, t, :] = x[:, :, t * self.stride:t * self.stride + 64].reshape(B, -1)
        xi = np.clip(np.round(cols / self.scales["patch"]), -127, 127).astype(np.int32)
        w = self.patch_w.reshape(self.d, -1).astype(np.int32)
        h = (xi @ w.T).astype(np.float32) * (self.patch_s * self.scales["patch"]) + self.patch_b
        h = h + self.pos[:T][None, :, :]

        # attention（post-LN：先注意力，再 LN1(h + attn)）
        hi = np.clip(np.round(h / self.scales["attn"]), -127, 127).astype(np.int32)
        qkv = (hi @ self.qkv_w.astype(np.int32).T).astype(np.float32) * (self.qkv_s * self.scales["attn"]) + self.qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.hd).transpose(0, 2, 1, 3)
        attn = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.hd), axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
        ci = np.clip(np.round(ctx / self.scales["ctx"]), -127, 127).astype(np.int32)
        out = (ci @ self.out_w.astype(np.int32).T).astype(np.float32) * (self.out_s * self.scales["ctx"]) + self.out_b
        h = layernorm(h + out, self.ln1_g, self.ln1_b)

        # FFN（post-LN）
        fi = np.clip(np.round(h / self.scales["ffn"]), -127, 127).astype(np.int32)
        f = (fi @ self.fc1_w.astype(np.int32).T).astype(np.float32) * (self.fc1_s * self.scales["ffn"]) + self.fc1_b
        f = gelu_tanh(f)
        gi = np.clip(np.round(f / self.scales["gelu"]), -127, 127).astype(np.int32)
        f = (gi @ self.fc2_w.astype(np.int32).T).astype(np.float32) * (self.fc2_s * self.scales["gelu"]) + self.fc2_b
        h = layernorm(h + f, self.lnf_g, self.lnf_b)          # 最后一层 LN
        pooled = h.mean(axis=1)
        pi = np.clip(np.round(pooled / self.scales["head"]), -127, 127).astype(np.int32)
        logits = (pi @ self.head_w.astype(np.int32).T).astype(np.float32) * (self.head_s * self.scales["head"]) + self.head_b
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"models\tiny_transformer_d16\tiny_transformer.pt")
    ap.add_argument("--out", default=r"models\tiny_transformer_d16")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    stride = int(cfg.get("stride", 32))
    T = (1024 - 64) // stride + 1
    qt = QuantTransformer(ck, d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], stride=stride, tokens=T)

    X, y = load_xyz()
    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    tr, va, te = split_random(y)
    wtr = tr[y[tr] > 0]
    Xcal = norm3(X[wtr][:1500], rms[wtr][:1500])
    st = qt.calibrate(Xcal)
    qt.scales = {k: max(1e-6, v / 127.0) for k, v in st.items()}
    print("[calib]", {k: round(v, 4) for k, v in st.items()}, "->", {k: round(v, 5) for k, v in qt.scales.items()})
    wte = te[y[te] > 0]
    Xte = norm3(X[wte], rms[wte])
    yte = y[wte] - 1
    probs = []
    for i in range(0, len(Xte), 256):
        probs.append(qt.forward(Xte[i:i + 256]))
    probs = np.concatenate(probs)
    pred = probs.argmax(1)
    acc = 100.0 * float((pred == yte).mean())
    print(f"[quant-numpy] 随机口径 wear3 = {acc:.2f}%")

    # 同权重 float 参考
    model = TinyTransformer(3, cfg["d"], cfg["heads"], cfg["ff"], cfg["layers"], 3, stride=stride)
    model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad():
        pf = model(torch.from_numpy(Xte).float()).argmax(1).numpy()
    print(f"[float-torch] 随机口径 wear3 = {100.0*float((pf==yte).mean()):.2f}%")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    n_par = sum(v.numel() for v in ck["state_dict"].values())
    # 生成 C 数组（int8 权重 + float 偏置/位置/LN 参数 + 尺度）
    def carr(name, arr, ctype, fmt):
        flat = arr.reshape(-1)
        lines = [f"const {ctype} {name}[{flat.size}] = {{"]
        for i in range(0, flat.size, 16):
            lines.append("  " + ", ".join(fmt % v for v in flat[i:i + 16]) + ",")
        lines.append("};")
        return "\n".join(lines)

    parts = ["/* 自动生成：tiny Transformer d=16（int8 权重 + float 激活） */",
             "#include \"transformer_model.h\"", ""]
    parts.append(carr("T_PATCH_W", qt.patch_w, "int8_t", "%d"))
    parts.append(carr("T_PATCH_B", qt.patch_b, "float", "%.7ef"))
    parts.append(carr("T_POS", qt.pos, "float", "%.7ef"))
    parts.append(carr("T_QKV_W", qt.qkv_w, "int8_t", "%d"))
    parts.append(carr("T_QKV_B", qt.qkv_b, "float", "%.7ef"))
    parts.append(carr("T_OUT_W", qt.out_w, "int8_t", "%d"))
    parts.append(carr("T_OUT_B", qt.out_b, "float", "%.7ef"))
    parts.append(carr("T_FC1_W", qt.fc1_w, "int8_t", "%d"))
    parts.append(carr("T_FC1_B", qt.fc1_b, "float", "%.7ef"))
    parts.append(carr("T_FC2_W", qt.fc2_w, "int8_t", "%d"))
    parts.append(carr("T_FC2_B", qt.fc2_b, "float", "%.7ef"))
    parts.append(carr("T_LN1_G", qt.ln1_g, "float", "%.7ef"))
    parts.append(carr("T_LN1_B", qt.ln1_b, "float", "%.7ef"))
    parts.append(carr("T_LN2_G", qt.ln2_g, "float", "%.7ef"))
    parts.append(carr("T_LN2_B", qt.ln2_b, "float", "%.7ef"))
    parts.append(carr("T_LNF_G", qt.lnf_g, "float", "%.7ef"))
    parts.append(carr("T_LNF_B", qt.lnf_b, "float", "%.7ef"))
    parts.append(carr("T_HEAD_W", qt.head_w, "int8_t", "%d"))
    parts.append(carr("T_HEAD_B", qt.head_b, "float", "%.7ef"))
    parts.append(
        "const float T_PATCH_S = %.9ef;\nconst float T_QKV_S = %.9ef;\nconst float T_OUT_S = %.9ef;\n"
        "const float T_FC1_S = %.9ef;\nconst float T_FC2_S = %.9ef;\nconst float T_HEAD_S = %.9ef;\n"
        % (qt.patch_s, qt.qkv_s, qt.out_s, qt.fc1_s, qt.fc2_s, qt.head_s))
    parts.append(
        "const int T_D = 16;\nconst int T_HEADS = 2;\nconst int T_HD = 8;\nconst int T_FF = 32;\nconst int T_TOKENS = %d;\n" % T)
    open(os.path.join(out, "transformer_model.c"), "w", encoding="utf-8").write("\n\n".join(parts) + "\n")

    header = """/* 自动生成：tiny Transformer d=16 接口 */
#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H
#include <stdint.h>

#define T_INPUT_C   3
#define T_INPUT_LEN 1024
#define T_NUM_CLASSES 3
#define T_TOKENS_N %d
#define T_PATCH_STRIDE %d
#define T_PARAM_COUNT %d
#define T_PATCH_X_S  %.9ef
#define T_ATTN_X_S   %.9ef
#define T_CTX_X_S    %.9ef
#define T_FFN_X_S    %.9ef
#define T_GELU_X_S   %.9ef
#define T_HEAD_X_S   %.9ef

extern const float T_PATCH_S, T_QKV_S, T_OUT_S, T_FC1_S, T_FC2_S, T_HEAD_S;
extern const int T_D, T_HEADS, T_HD, T_FF, T_TOKENS;

int transformer_forward(const float *input, float *probs);

#endif
""" % (T, stride, n_par, qt.scales["patch"], qt.scales["attn"], qt.scales["ctx"], qt.scales["ffn"], qt.scales["gelu"], qt.scales["head"])
    open(os.path.join(out, "transformer_model.h"), "w", encoding="utf-8").write(header)

    np.savez_compressed(os.path.join(out, "parity_transformer.npz"),
                        features=Xte[:100].astype(np.float32), labels=yte[:100],
                        probs=probs[:100].astype(np.float32))
    json.dump({"acc_quant_numpy": acc, "acc_float": 100.0 * float((pf == yte).mean()),
               "params": n_par, "d": cfg["d"], "heads": cfg["heads"], "ff": cfg["ff"]},
              open(os.path.join(out, "metrics_transformer.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[saved]", os.path.join(out, "transformer_model.c/h"))


if __name__ == "__main__":
    main()
