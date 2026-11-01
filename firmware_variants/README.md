# 固件版本说明

根目录只保留两个最常用的固件；其它版本放在本目录，按需 `cd` 进去 `build.bat` / `tools\flash.bat`。

| 目录 | 用途 | 关键指标 |
|---|---|---|
| `../firmware/` | 单轴在线 ADC（PA0） | 100.0 + 17.07 = **117.1 ms/窗** |
| `../firmware_xyz_relabel/` | **当前演示**：三轴 DAC 分时回放 + 修正标签模型 | 300.0 + 31.42 = **331.4 ms/窗** |
| `xyz_replay/` | 三轴回放（原始标签模型） | 300.0 + 31.4 = **331.4 ms/窗** |
| `xyz_sync/` | 三轴同步采集（PA0/PA1/PA2，接真实传感器用） | 100.1 + 31.39 = **131.5 ms/窗**（已编译） |
| `xyz_transformer/` | 轻量 Transformer 实验（8-token） | 59 ms/窗（精度低于 CNN） |

## 烧录示例

```powershell
cd firmware_variants\xyz_sync
.\build.bat
.\tools\flash.bat
```

> 每个固件目录内部都有 `src/ csrc/ cmsis/ models/ linker/ tools/ build.bat`，路径都是相对自身，
> 所以移动目录不影响编译；烧录前确认 DAP 已连接、串口没有被看板占用。
