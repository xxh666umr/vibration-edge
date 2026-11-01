# models 说明

## 当前使用（根目录）

| 目录 | 内容 | 用途 |
|---|---|---|
| `gate3/` | 单轴模型权重、int8 C 数组、metrics、parity | 单轴固件 `firmware/` |
| `gate3_xyz/` | 三轴模型（原始标签） | 三轴回放固件 `firmware_variants/xyz_replay/` |
| `relabel_xyz_t90/` | 三轴模型（信号重标标签） | 当前演示固件 `firmware_xyz_relabel/` |
| `tiny_transformer_d16/` | 31-token 轻量 Transformer（d=16） | 实验记录，未部署 |
| `tiny_transformer_t16/` | 16-token Transformer | 实验记录，未部署 |
| `tiny_transformer_t8/` | 8-token Transformer（板端 59 ms） | 实验固件 `firmware_variants/xyz_transformer/` |
| `_experiments/` | 早期调参产物（block / g1_s8 / g2_wide / 各种 relabel 变体） | 归档，不参与演示 |

## 说明

- 每个模型目录一般包含 `*.pt`（PyTorch）、`model_vib.c/h`（int8 C 数组）、`metrics*.json`；
- 训练脚本默认输出到 `models\<name>`，重跑时注意不要覆盖当前模型；
- 归档的模型仍保留，用于追溯早期实验；README 只列当前使用项。
