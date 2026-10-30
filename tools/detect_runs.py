# -*- coding: utf-8 -*-
"""检测每个类文件内部的“试验段/刀具”结构（用于按段划分的严格评测）。

方法：对每个类文件提取窗口特征（log RMS、频谱质心、峭度、频带能量），
标准化后用二分类分割（binary segmentation）找突变点，惩罚项用 BIC 近似。
输出每类的段数、段长、每段特征均值（便于判断是否按刀具/工况分组）。
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\lenovo\Desktop\vibration-edge\tools")
from transition_experiment import features_1ch, load_x  # noqa: E402

NAMES = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]


def seg_cost(F):
    if len(F) < 2:
        return 0.0
    # 多变量段代价：n * log(det(cov + eps))
    n, d = F.shape
    cov = np.cov(F.T) + np.eye(d) * 1e-6
    sign, logdet = np.linalg.slogdet(cov)
    return float(n * logdet) if sign > 0 else float(n * np.log(1e-6 + F.var(0).sum()))


def binary_seg(F, max_seg=20, min_len=300):
    cuts = [0, len(F)]
    # 惩罚项：每增加一段，BIC 惩罚约 0.5*(d + d(d+1)/2)*log(n)
    n, d = F.shape
    pen = 0.5 * (d + d * (d + 1) / 2) * np.log(n) * 0.5
    for _ in range(max_seg - 1):
        best = None
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 2 * min_len:
                continue
            base = seg_cost(F[a:b])
            step = max(50, (b - a) // 20)
            for t in range(a + min_len, b - min_len + 1, step):
                gain = base - seg_cost(F[a:t]) - seg_cost(F[t:b])
                if best is None or gain > best[0]:
                    best = (gain, t)
        if best is None or best[0] < pen:
            break
        cuts = sorted(cuts + [best[1]])
    return cuts


def main():
    X, y = load_x()
    F = features_1ch(X)
    for c in range(4):
        idx = np.where(y == c)[0]
        Fc = F[idx]
        # 每类内部标准化
        Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-9)
        cuts = binary_seg(Fc)
        segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        print(f"\n=== {NAMES[c]} n={len(idx)} 段数={len(segs)} ===")
        for k, (a, b) in enumerate(segs):
            m = Fc[a:b].mean(0)
            print(f"  seg{k:2d}: [{a:6d},{b:6d}) len={b-a:6d}  logRMS={m[0]:+.2f} centroid={m[1]:+.2f} kurt={m[2]:+.2f}")


if __name__ == "__main__":
    main()
