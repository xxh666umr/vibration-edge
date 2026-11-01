# vibration-edge：工业振动（刀具磨损）识别 · STM32F103 边缘部署

把 SDU-QIT 刀具磨损振动数据（4 类：空闲 / 初期磨损 / 稳定磨损 / 加速磨损）训练成**很小的 int8 CNN**，
导出为纯 C 数组，在 **STM32F103ZET6** 上完成**在线 ADC 采集 → 能量门限 → int8 推理 → 串口回传 → Web 看板**。

## 结果速览

| 方案 | 参数量 | 准确率（随机划分，论文口径） | 板端实测 |
|---|---|---|---|
| 单轴（x）两级方案：RMS 门限 + 3 类 CNN | 8,435 | 4 类 **96.74%**（int8） | 采集 100.0 ms + 推理 17.07 ms = **117.1 ms/窗** |
| 三轴（x/y/z）两级方案 | 10,483 | 4 类 **99.00%**（int8；float 99.02%） | 回放采集 300.0 ms + 推理 31.42 ms = **331.4 ms/窗** |
| 三轴同步采集版（待传感器） | 10,483 | 同上 | 100.1 ms + 31.39 ms = **131.5 ms/窗**（已编译） |

严格口径（按时间划分、测试最后 10%）会掉到 42.8% / 45.5%：原因已定位为**阶段边界标签与信号不一致**
（每个阶段文件末段已进入下一阶段幅值区间；三篇参考文献均未给出阶段边界规则）。
严格口径下所有错误都是**相邻阶段混淆**（跨档错误 0.1%），信号重标后严格口径恢复到 **95.99%**。
详见 `docs/`。

## 目录结构

```
vibration-edge/
├── start_demo.bat              # 一键启动 Web 看板（当前演示）
├── demo/                       # 串口解析 + Web 看板（pyserial）
├── firmware/                   # ★ 单轴固件：在线 ADC + 门限 + CNN（117.1 ms）
├── firmware_xyz_relabel/       # ★ 当前演示固件：三轴 DAC 回放 + 修正标签模型（331.4 ms）
├── firmware_variants/          # 其它版本，按需使用
│   ├── xyz_replay/             #   三轴回放（原始标签版）
│   ├── xyz_sync/              #   三轴同步采集（PA0/PA1/PA2，131.5 ms，接传感器用）
│   └── xyz_transformer/       #   轻量 Transformer 实验固件（8-token，59 ms）
├── models/                     # 训练产物 / 导出的 C 数组
│   ├── gate3/                  #   单轴最终模型
│   ├── gate3_xyz/              #   三轴最终模型（原始标签）
│   ├── relabel_xyz_t90/        #   三轴修正标签模型（当前演示）
│   ├── tiny_transformer_*/     #   轻量 Transformer 实验（d16 / t16 / t8）
│   └── _experiments/           #   历史调参产物（归档，不参与演示）
├── tools/                      # 训练 / 量化 / 对齐 / 分析脚本
└── docs/                       # 架构、演示、口径说明（本仓库内）
```

**记住两个路径就够了**：演示烧 `firmware_xyz_relabel/`，讲单轴烧 `firmware/`。

## 快速开始

### 0. 环境

- Python 3.10+：`numpy torch pyserial`（训练）、`pyserial`（看板）；
- 数据：SDU-QIT 数据集（山东大学×齐鲁理工学院，<https://www.qlit.edu.cn/datasets/>），
  解压后 `trainData_{x,y,z}_{0..3}.npy` 放到本地数据目录，并在 `tools/train_vib*.py` 的 `DATA_DIR` 中配置路径（数据不随仓库分发）；
- 固件：ARM GNU Toolchain 14.2 + OpenOCD + CMSIS-DAP 调试器。

### 1. 训练 + 量化 + 导出 C（可选，仓库已带好模型）

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
# 当前演示：三轴修正标签回放
cd firmware_xyz_relabel
.\build.bat
.\tools\flash.bat

# 单轴（117.1 ms）
cd ..\firmware
.\build.bat
.\tools\flash.bat

# 三轴同步采集（接传感器时用，131.5 ms）
cd ..\firmware_variants\xyz_sync
.\build.bat
.\tools\flash.bat
```

> PowerShell 里必须写 `.\build.bat`（当前目录下的程序要加 `.\`）；用 cmd 或双击则直接 `build.bat`。

串口 USART1（PA9/PA10，115200 8N1）输出每窗结果与 DWT 实测时延。
当前演示固件为离线回放（板内 DAC 回放真实样本）；把 **PA4→PA0** 短接可看到 ADC 通道与回放一致。

### 4. 看板

双击 `start_demo.bat`（或 `python demo\server.py --port COM8 --http-port 8766`），浏览器打开
<http://127.0.0.1:8766>；无板子时 `python demo\server.py --simulate`。

## 说明

- 数据集许可遵循 SDU-QIT 原始协议，数据不随仓库分发；
- 汇报材料、答辩问答、看板截图等"阅读资料"在桌面 `资料与汇报\振动识别\`，不属于本代码仓库；
- 严格口径（按时间块）结果、逐时间段分析与原因说明见 `docs/`；
- 旧的调参模型已归档到 `models/_experiments/`，旧固件在 `firmware_variants/`，根目录只保留当前演示与主固件。
