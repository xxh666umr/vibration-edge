# 工业振动识别 Web 实时看板

板端（STM32F103ZET6）通过 USART1/CH340 把每个窗口的识别结果发回 PC，
本目录的 `server.py` 解析串口帧并提供网页实时显示。

## 启动

双击项目根目录的 `start_demo.bat`，或：

```powershell
python demo\server.py --port COM6      # 接板子（不填 --port 会自动找 CH340）
python demo\server.py --simulate       # 无板子演示
```

浏览器打开 <http://127.0.0.1:8766>（默认端口 8766，可用 `--http-port` 修改）。

## 页面内容

- **当前判定**：空闲（RMS 门限）/ 三种磨损（CNN），来源标注为「ADC 通道」或「内置回放」
- **时延**：采集 ms（默认 100 ms/1024 点 @10.24 kHz）、推理 ms、总计 ms/窗，以及最近 30 窗柱状图
- **磨损类别概率**：3 类 CNN 概率条；空闲窗口显示最近一次磨损窗口的概率
- **板端信息**：串口、链路、窗口数、ADC 超时采样数、更新时间、模型信息
- **最近识别记录**：时间、回放样本类别、ADC 判定、CNN 预测、置信度、采集/推理/总计

## 串口帧格式（固件输出的要点）

```
--- window: replay class 1 (initial_wear) ---
[acq  ] ... (100.00 ms), 1024 samples @ 10240 Hz, adc_timeouts=0
[adc  ] mean=... rms_ac=... -> idling | wear, pred=0 (initial_wear) infer=... cycles (... ms)
[replay] mean=... rms_ac=... -> idling | wear, pred=0 (initial_wear) infer=... cycles (... ms)
    class 0 (initial_wear): 0.9598
    class 1 (steady_wear): 0.0171
    class 2 (accelerating_wear): 0.0231
[window] acq 100.0 ms + infer 17.07 ms = 117.1 ms/decision (window 100 ms)
```

## 依赖

仅 `pyserial`（`pip install pyserial`），HTTP 服务用 Python 标准库，无 Web 框架。
