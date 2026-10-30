# -*- coding: utf-8 -*-
"""三轴模型 C vs numpy 逐样本对齐测试。"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_vib import quantize_sym, quantize_vib, quantized_forward  # noqa: E402
from train_vib import VibNet, split_random  # noqa: E402
from train_vib3 import load_xyz, norm3  # noqa: E402

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "edge-inference", "csrc")


def vcvars():
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
    ap.add_argument("--model-dir", default=r"models\gate3_xyz")
    ap.add_argument("--ckpt", default=r"models\gate3_xyz\vibnet_gate3_xyz.pt")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    md = os.path.abspath(args.model_dir)

    import torch
    import torch.nn as nn
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = VibNet(cfg["c1"], cfg["c2"], cfg["c3"], cfg["k1"], cfg["s1"], nc=3)
    model.conv1 = nn.Conv1d(3, cfg["c1"], cfg["k1"], cfg["s1"])
    model.load_state_dict(ck["state_dict"]); model.eval()

    calib_path = os.path.join(md, "calib_features.npy")
    if os.path.exists(calib_path):
        calib = np.load(calib_path)
        print(f"[calib] 使用模型目录保存的校准特征 {calib.shape}", flush=True)
        q = quantize_vib(model, calib)
    else:
        X, y = load_xyz()
        rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
        tr, va, te = split_random(y)
        m_tr = np.zeros(len(y), bool); m_tr[tr] = True
        wtr = np.where(m_tr & (y > 0))[0]
        q = quantize_vib(model, norm3(X[wtr], rms[wtr])[:1500])
    q["classes"] = ck["class_names"]

    data = np.load(os.path.join(md, "parity_vib.npz"))
    feats = data["features"][:args.n]          # (N,3,1024)
    pred_np, probs_np, _ = quantized_forward(q, feats)

    work = tempfile.mkdtemp(prefix="vib3_parity_")
    for fn in ("model_vib.c", "model_vib.h"):
        shutil.copy(os.path.join(md, fn), work)
    for fn in ("nn.c", "nn.h"):
        shutil.copy(os.path.join(CSRC, fn), work)
    xq = quantize_sym(feats, q["input"]["scale"]).astype(np.int8)
    feat_bin = os.path.join(work, "features_q.bin")
    xq.tofile(feat_bin)
    probe = os.path.join(work, "probe.c")
    open(probe, "w", encoding="utf-8").write("""#include <stdio.h>
#include "model_vib.h"
int main(int argc, char **argv) {
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
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
    exe = os.path.join(work, "probe.exe")
    pbin = os.path.join(work, "probs.bin")
    bat = os.path.join(work, "build.bat")
    open(bat, "w", encoding="ascii").write(
        '@echo off\ncall "%s" >nul 2>&1\n'
        'cl /nologo /O2 /utf-8 /W0 /I"%s" "%s" "%s\\model_vib.c" "%s\\nn.c" /Fe:"%s"\n'
        'if errorlevel 1 exit /b 1\n"%s" "%s" "%s"\n' % (vcvars(), work, probe, work, work, exe, exe, feat_bin, pbin))
    r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="ignore")
    if r.returncode != 0 or not os.path.exists(pbin):
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit("build failed")
    probs_c = np.fromfile(pbin, dtype=np.float32).reshape(len(feats), -1)
    same = int((probs_c.argmax(1) == pred_np).sum())
    md_ = float(np.abs(probs_c - probs_np).max())
    print(f"[xyz] 样本 {len(feats)}, 输入 3x1024")
    print(f"[argmax] C vs numpy 一致 {same}/{len(feats)} = {100.0*same/len(feats):.2f}%")
    print(f"[probs ] 最大绝对误差 {md_:.3e}")
    print("PASS" if same == len(feats) and md_ < 1e-4 else "FAIL")
    if same != len(feats) or md_ >= 1e-4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
