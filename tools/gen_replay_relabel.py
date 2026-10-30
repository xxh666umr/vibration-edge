# -*- coding: utf-8 -*-
"""按“信号重标标签”重新挑选回放样本（让回放演示与修正标签一致）。

每个回放类别选一个重标标签匹配的窗口：
  0 idling      -> 空闲文件任意窗口（门限判空闲）
  1 initial     -> initial 文件中重标为 initial 的窗口
  2 steady      -> 重标为 steady 的窗口（initial 后段或 steady 前段）
  3 accelerating-> 重标为 accelerating 的窗口（steady 后段或 accelerating）
生成 firmware_xyz_relabel/models/replay_xyz.h（4 类 × 3 轴 × 1024 的 DAC 码）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\lenovo\Desktop\vibration-edge\tools")
from transition_experiment import features_3ch  # noqa: E402
from train_vib3 import load_xyz  # noqa: E402
from relabel_experiment import relabel  # noqa: E402

NAMES = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]
AXES = ["x", "y", "z"]
GAIN, VREF = 5.0, 3.3
OUT = r"C:\Users\lenovo\Desktop\vibration-edge\firmware_xyz_relabel\models\replay_xyz.h"


def main():
    X, y = load_xyz()
    feats = features_3ch(X)
    lab, _, _ = relabel(y, feats, 1.15, use_transition=False)

    picks = {}
    picks[0] = int(np.where(y == 0)[0][100])
    for c, tag in ((1, 0), (2, 1), (3, 2)):
        idx = np.where((y == c) & (lab == tag))[0]
        # 取该标签段的中部，避免边界
        picks[c] = int(idx[len(idx) // 2])
    print("回放样本 index:", picks)
    for c in range(4):
        print(f"  class {c} ({NAMES[c]}): index={picks[c]} 原始标签={int(y[picks[c]])} 重标标签={int(lab[picks[c]])}")

    lines = ["/* 自动生成：按信号重标标签挑选的回放样本（4 类 × 三轴 × 1024 DAC 码）。 */",
             "#ifndef REPLAY_XYZ_H", "#define REPLAY_XYZ_H", "#include <stdint.h>",
             "#define REPLAY_N 4", "#define REPLAY_AXES 3", "#define REPLAY_LEN 1024",
             "static const int16_t REPLAY_XYZ[REPLAY_N][REPLAY_AXES][REPLAY_LEN] = {"]
    for c in range(4):
        lines.append(f"  {{  /* {NAMES[c]} */")
        for k in range(3):
            x = X[picks[c], k]
            v = np.clip(x * GAIN, -1.6, 1.6)
            codes = np.clip(np.round(2048 + v / VREF * 4095), 0, 4095).astype(np.int64)
            rows = [", ".join(str(int(t)) for t in codes[i:i + 32]) for i in range(0, 1024, 32)]
            lines.append(f"    {{  /* axis {AXES[k]} */")
            lines.append("      " + ",\n      ".join(rows))
            lines.append("    }," if k < 2 else "    }")
        lines.append("  }," if c < 3 else "  }")
    lines += ["};",
              'static const char *REPLAY_NAME[REPLAY_N] = {"idling", "initial_wear", "steady_wear", "accelerating_wear"};',
              "#endif"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
