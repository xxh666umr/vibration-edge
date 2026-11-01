# tools 说明

| 类别 | 脚本 | 用途 |
|---|---|---|
| 训练 | `train_vib.py` | 通用 VibNet（单轴 / 三轴、随机或时间块划分） |
| | `train_vib2.py` | 单轴两级方案（门限 + 3 类 CNN） |
| | `train_vib3.py` | 三轴两级方案 |
| 量化 | `quantize_vib.py` / `quantize_vib2.py` / `quantize_vib3.py` | float → int8，导出 C 数组 |
| | `export_relabel_c.py` | 重标模型导出（当前演示模型） |
| 对齐 | `test_parity_vib.py` / `test_parity_vib3.py` | C 实现 vs numpy 逐样本对齐 |
| | `demo_parity.bat` / `demo_train_quick.bat` | 演示用一键脚本 |
| 研究实验 | `transition_experiment.py` | 过渡态/时间块诊断 |
| | `relabel_experiment.py` | 信号驱动的标签重标 |
| | `train_twmnet_block.py` | TWM-Net 严格口径对照 |
| | `tiny_transformer.py` | 轻量 Transformer 训练（8/16/31 token） |
| | `export_transformer_c.py` | 轻量 Transformer 导出 C |
| | `test_parity_transformer.py` | Transformer C vs numpy 对齐 |
| 板端数据 | `gen_replay_relabel.py` | 生成 DAC 回放数组 `replay_xyz.h` |
| 分析 | `detect_runs.py` | 回放/日志分析 |

## 数据路径

`train_vib*.py` / `relabel_experiment.py` 里的 `DATA_DIR` 需要指向本地 SDU-QIT 数据目录（数据不随仓库分发）。
