# 现场演示流程（板端 CNN）

## 准备

1. 板子电源打开（PWR 灯亮）；
2. CH340 USB 插好（当前为 **COM8**，以设备管理器为准）；
3. DAP + SWD 插好（仅烧录时需要）；
4. **PA4 ↔ PA0 杜邦线接好**（DAC 回放 → ADC 采样）；
5. 关闭占用串口的程序（同一串口只能被一个程序打开）。

## 启动

```powershell
cd C:\Users\lenovo\Desktop\vibration-edge
.\start_demo.bat
```

浏览器打开 <http://127.0.0.1:8766>（注意不是 8765，8765 是调制识别项目）。

## 看板上应看到

- 当前判定 `accelerating_wear` 等（约 95~98%）；
- 来源 **ADC 通道**；ADC 判定与回放一致；
- 时延 **300.0 + 31.4 = 331.4 ms/窗**；
- 历史记录里 initial / steady / accelerating 三档轮换，空闲由门限判（推理 0.00 ms）。

## 切换单轴（老师要看时）

```powershell
# 先关看板释放 COM8
cd C:\Users\lenovo\Desktop\vibration-edge\firmware
.\build.bat
.\tools\flash.bat
# 重新开看板，应显示 100.0 + 17.07 = 117.1 ms/窗
```

切回三轴：

```powershell
cd ..\firmware_xyz_relabel
.\build.bat
.\tools\flash.bat
```

## 排错

| 现象 | 处理 |
|---|---|
| 看板无数据 | 确认地址是 8766、板子电源、COM8；刷新页面 |
| 串口 Access denied | 看板/串口工具二选一，先关掉另一个 |
| 看板显示"无信号" | PA4-PA0 杜邦线没接或松了 |
| 板子无输出 | 按 RESET；检查电源开关；确认烧的是 `firmware_xyz_relabel` |
