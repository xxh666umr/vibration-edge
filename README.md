# vibration-edge：工业振动（刀具磨损）识别 · STM32F103 边缘部署

把 SDU-QIT 刀具磨损振动数据（4 类：空闲 / 初期磨损 / 稳定磨损 / 加速磨损）训练成**很小的 int8 CNN**，
导出为纯 C 数组，在 **STM32F103ZET6** 上完成**在线 ADC 采集 → 能量门限 → int8 推理 → 串口回传 → Web 看板**。

## 结果速览

| 方案 | 参数量 | 准确率（随机划分，论文口径） | 板端实测 |
|---|---|---|---|
| 单轴（x）两级方案：RMS 门限 + 3 类 CNN | 8,435 | 4 类 **96.74%**（int8） | 采集 100.0 ms + 推理 17.07 ms = **117.1 ms/窗** |
| 三轴（x/y/z）两级方案 | 10,483 | 4 类 **99.00%**（int8） | 采集 300.0 ms + 推理 31.42 ms = **331.4 ms/窗** |

严格口径（按时间划分、测试最后 10%）会掉到 42.8% / 45.5%：原因已定位为**阶段边界标签与信号不一致**
（每个阶段文件末段已进入下一阶段幅值区间，三篇参考文献均未给出阶段边界规则）。
严格口径下所有错误都是**相邻阶段混淆**（跨档错误 0），详见 `docs/` 中的说明。

## 目录结构

```
vibration-edge/
├── start_demo.bat          # 一键启动 Web 看板
├── demo/                   # 串口解析 + Web 实时看板（仅依赖 pyserial）
│   ├── server.py
│   ├── static/index.html
│   └── README.md
├── tools/                  # 训练 / 量化 / 对齐 / 分析脚本
│   ├── train_vib.py        # 单模型（随机/时间块划分）
│   ├── train_vib2.py       # 单轴两级方案（门限 + 3 类 CNN）
│   ├── train_vib3.py       # 三轴两级方案
│   ├── quantize_vib.py     # int8 量化 + C 导出（通用）
│   ├── quantize_vib2.py    # 单轴两级方案量化
│   ├── quantize_vib3.py    # 三轴量化
│   ├── test_parity_vib.py  # 单轴 C vs numpy 逐样本对齐
│   ├── test_parity_vib3.py # 三轴对齐
│   ├── transition_experiment.py  # 过渡态实验
│   ├── relabel_experiment.py     # 信号重标实验
│   └── train_twmnet_block.py     # TWM-Net 严格口径对照
├── models/                 # 训练产物与导出的 C 数组
│   ├── gate3/              # 单轴最终模型（model_vib.c/h、metrics、parity）
│   └── gate3_xyz/          # 三轴最终模型
├── firmware/               # 单轴固件（在线 ADC + 门限 + CNN）
├── firmware_xyz/           # 三轴固件（DAC 分时回放三轴 + ADC 3×1024）
└── docs/                   # 技术说明（汇报材料等阅读资料不在此仓库）
```

## 快速开始

### 0. 环境

- Python 3.10+，`numpy pandas pyarrow torch`（训练）、`pyserial`（看板）
- 数据：SDU-QIT 数据集（山东大学×齐鲁理工学院，<https://www.qlit.edu.cn/datasets/>），
  解压后 `trainData_{x,y,z}_{0..3}.npy` 放到本地数据目录，并在 `tools/train_vib*.py` 的 `DATA_DIR` 中配置路径（数据不随仓库分发）。
- 固件：ARM GNU Toolchain 14.2 + OpenOCD + CMSIS-DAP 调试器。

### 1. 训练 + 量化 + 导出 C

```powershell
python tools\train_vib2.py --split random --epochs 80 --out models\gate3       # 单轴
python tools\quantize_vib2.py --ckpt models\gate3\vibnet_gate3.pt --out models\gate3
python tools\train_vib3.py --split random --epochs 80 --out models\gate3_xyz   # 三轴
python tools\quantize_vib3.py --ckpt models\gate3_xyz\vibnet_gate3_xyz.pt --out models\gate3_xyz
```

### 2. 对齐验证（C vs numpy）

```powershell
python tools\test_parity_vib.py  --model-dir models\gate3     --ckpt models\gate3\vibnet_gate3.pt
python tools\test_parity_vib3.py --model-dir models\gate3_xyz --ckpt models\gate3_xyz\vibnet_gate3_xyz.pt
```

### 3. 编译 / 烧录固件

```powershell
cd firmware        # 单轴
build.bat
tools\flash.bat

cd ..\firmware_xyz # 三轴
build.bat
tools\flash.bat
```

串口 USART1（PA9/PA10，115200 8N1）输出每窗结果与 DWT 实测时延。
当前固件为离线回放演示（板内 DAC 回放真实样本）；把 **PA4→PA0** 短接可看到 ADC 通道与回放一致。

### 4. 看板

双击 `start_demo.bat`（或 `python demo\server.py --port COM6`），浏览器打开 <http://127.0.0.1:8766>；
无板子时 `python demo\server.py --simulate`。

## 说明

- 数据集许可遵循 SDU-QIT 原始协议，数据不随仓库分发；
- 汇报材料、进展文档、看板截图等"阅读资料"在桌面 `资料与汇报\振动识别\`，不属于本代码仓库；
- 严格口径（按时间块）结果、逐时间段分析与原因说明见 `docs/` 及桌面资料。
