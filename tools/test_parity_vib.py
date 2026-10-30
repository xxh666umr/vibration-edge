r"""C 推理 vs numpy 整数参考：逐样本对齐测试（两层 MCU 方案：门限 + 3 类 CNN）。

用法：
    python tools/test_parity_vib.py --model-dir models\gate3 --ckpt models\gate3\vibnet_gate3.pt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_vib import quantize_vib, quantized_forward, quantize_sym  # noqa: E402
from train_vib import VibNet, split_random  # noqa: E402
from train_vib2 import load_xyzr, normalize_wave  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "edge-inference")
CSRC = os.path.join(REPO, "csrc")


def find_vcvars64():
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    for root in roots:
        for ed in ("Community", "Professional", "Enterprise", "BuildTools"):
            p = os.path.join(root, "Microsoft Visual Studio", "2022", ed,
                             "VC", "Auxiliary", "Build", "vcvars64.bat")
            if os.path.exists(p):
                return p
    raise FileNotFoundError("vcvars64.bat not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"models\gate3")
    ap.add_argument("--ckpt", default=r"models\gate3\vibnet_gate3.pt")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    model_dir = os.path.abspath(args.model_dir)

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    classes = ck["class_names"]
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], nc=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X, y, rms = load_xyzr("x")
    tr, va, te = split_random(y)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True
    wtr = np.where(m_tr & (y > 0))[0]
    Xw_tr = normalize_wave(X[wtr], rms[wtr])
    q = quantize_vib(model, Xw_tr[:2000][:, None, :])
    q["classes"] = classes

    data = np.load(os.path.join(model_dir, "parity_vib.npz"))
    feats = data["features"][:args.n]          # (N,1,1024) normalized
    labels = data["labels"][:args.n]
    pred_np, probs_np, _ = quantized_forward(q, feats)

    import shutil
    work = tempfile.mkdtemp(prefix="vib_parity_")
    # 拷贝到纯 ASCII 临时目录，避免中文路径在 cmd/bat 里的编码问题
    for fn in ("model_vib.c", "model_vib.h"):
        shutil.copy(os.path.join(model_dir, fn), os.path.join(work, fn))
    for fn in ("nn.c", "nn.h"):
        shutil.copy(os.path.join(CSRC, fn), os.path.join(work, fn))
    xq = quantize_sym(feats, q["input"]["scale"]).astype(np.int8)
    feat_bin = os.path.join(work, "features_q.bin")
    xq.tofile(feat_bin)
    probe_c = os.path.join(work, "probe_vib.c")
    open(probe_c, "w", encoding="utf-8").write("""#include <stdio.h>
#include <stdlib.h>
#include "model_vib.h"
int main(int argc, char **argv) {
    FILE *fi = fopen(argv[1], "rb");
    FILE *fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 1;
    int8_t x[MODEL_INPUT_C * MODEL_INPUT_LEN];
    float p[MODEL_NUM_CLASSES];
    while (fread(x, 1, sizeof(x), fi) == sizeof(x)) {
        model_forward(x, p);
        fwrite(p, sizeof(float), MODEL_NUM_CLASSES, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
""")
    exe = os.path.join(work, "probe_vib.exe")
    probs_bin = os.path.join(work, "probs_c.bin")
    build = os.path.join(work, "build.bat")
    open(build, "w", encoding="ascii").write(
        f'@echo off\ncall "{find_vcvars64()}" >nul 2>&1\n'
        f'cl /nologo /O2 /utf-8 /W0 /I"{work}" '
        f'"{probe_c}" "{work}\\model_vib.c" "{work}\\nn.c" /Fe:"{exe}"\n'
        f'if errorlevel 1 exit /b 1\n"{exe}" "{feat_bin}" "{probs_bin}"\n')
    r = subprocess.run(["cmd", "/c", build], capture_output=True, text=True, errors="ignore")
    if r.returncode != 0 or not os.path.exists(probs_bin):
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("build/run failed")

    probs_c = np.fromfile(probs_bin, dtype=np.float32).reshape(len(feats), -1)
    argmax_c = probs_c.argmax(1)
    same = int((argmax_c == pred_np).sum())
    maxdiff = float(np.abs(probs_c - probs_np).max())
    print(f"样本: {len(feats)} 条, 特征 1x1024 (归一化后)")
    print(f"[argmax] C vs numpy 一致 {same}/{len(feats)} = {100.0*same/len(feats):.2f}%")
    print(f"[probs ] 最大绝对误差 {maxdiff:.3e}（阈值 1e-4）")
    ok = same == len(feats) and maxdiff < 1e-4
    print("PASS: C 推理与 numpy 整数参考逐样本一致" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
