# -*- coding: utf-8 -*-
"""C 版 tiny Transformer vs numpy 量化实现逐样本对齐。"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

FW = r"C:\Users\lenovo\Desktop\vibration-edge\firmware_xyz_transformer"
MODEL = r"C:\Users\lenovo\Desktop\vibration-edge\models\tiny_transformer_d16"


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
    data = np.load(os.path.join(MODEL, "parity_transformer.npz"))
    feats = data["features"].astype(np.float32)
    probs_np = data["probs"].astype(np.float32)
    labels = data["labels"]
    n = len(feats)
    work = tempfile.mkdtemp(prefix="tf_parity_")
    for f in ("transformer.c", "transformer.h"):
        shutil.copy(os.path.join(FW, "csrc", f), work)
    for f in ("transformer_model.c", "transformer_model.h"):
        shutil.copy(os.path.join(FW, "models", f), work)
    feats.tofile(os.path.join(work, "features.bin"))
    open(os.path.join(work, "probe.c"), "w", encoding="utf-8").write("""#include <stdio.h>
#include "transformer.h"
int main(int argc, char **argv) {
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 1;
    float x[3 * 1024];
    float p[3];
    while (fread(x, sizeof(float), 3 * 1024, fi) == 3 * 1024) {
        transformer_forward(x, p);
        fwrite(p, sizeof(float), 3, fo);
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
        'cl /nologo /O2 /utf-8 /W0 /I"%s" "%s\\probe.c" "%s\\transformer.c" "%s\\transformer_model.c" /Fe:"%s"\n'
        'if errorlevel 1 exit /b 1\n"%s" "%s\\features.bin" "%s"\n'
        % (vcvars(), work, work, work, work, exe, exe, work, pbin))
    r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="ignore")
    if r.returncode != 0 or not os.path.exists(pbin):
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit("build failed")
    probs_c = np.fromfile(pbin, dtype=np.float32).reshape(n, 3)
    same = int((probs_c.argmax(1) == probs_np.argmax(1)).sum())
    md = float(np.abs(probs_c - probs_np).max())
    print(f"[C vs numpy] 样本 {n}，argmax 一致 {same}/{n} = {100.0*same/n:.2f}%")
    print(f"[C vs numpy] 概率最大绝对误差 {md:.3e}")
    print(f"[C] 对标签准确率 {100.0*float((probs_c.argmax(1)==labels).mean()):.2f}%")
    print("PASS" if same == n and md < 2e-3 else "FAIL")
    if same != n or md >= 2e-3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
