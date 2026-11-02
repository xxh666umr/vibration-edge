#!/usr/bin/env python3
"""工业振动（刀具磨损 4 分类）实时看板：串口 -> 浏览器。

用法：
    python demo/server.py --port COM6            # 接板子（默认自动找 CH340）
    python demo/server.py --simulate             # 无板子演示
    python demo/server.py --no-browser --http-port 8766

页面：http://127.0.0.1:8766
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in (os.path.join(ROOT, ".deps"), os.environ.get("EDGE_DEPS", "")):
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

CLASSES = ["initial_wear", "steady_wear", "accelerating_wear"]
CLASS_CN = {"idling": "空闲", "initial_wear": "初期磨损",
            "steady_wear": "稳定磨损", "accelerating_wear": "加速磨损"}

DATA_DIR = os.environ.get("VIB_DATA_DIR",
                          r"C:\Users\lenovo\Desktop\工业语义通信Demo项目\data\Toolweardata")
SAMPLE_NAMES = ["idling", "initial_wear", "steady_wear", "accelerating_wear"]
SAMPLE_CN = ["空闲", "初期磨损", "稳定磨损", "加速磨损"]
SAMPLE_COUNTS = [11165, 6588, 57096, 15372]
SAMPLE_OFFSETS = [0, 11165, 11165 + 6588, 11165 + 6588 + 57096]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENOCD = os.environ.get(
    "OPENOCD",
    r"C:\Users\lenovo\AppData\Local\Microsoft\WinGet\Packages\xpack-dev-tools.openocd-xpack_Microsoft.Winget.Source_8wekyb3d8bbwe\xpack-openocd-0.12.0-7\bin\openocd.exe")
FIRMWARES = {
    "cnn": {"label": "CNN v3", "elf": os.path.join(ROOT, "firmware_xyz_relabel", "build", "firmware.elf")},
    "transformer": {"label": "Transformer tiny",
                    "elf": os.path.join(ROOT, "firmware_variants", "xyz_transformer", "build", "firmware.elf")},
}


def sample_payload(cls: int, idx=None):
    """读取 SDU-QIT 本地 npy，返回一个窗口的降采样波形和标签。"""
    import numpy as np
    cls = max(0, min(3, int(cls)))
    path = os.path.join(DATA_DIR, f"trainData_x_{cls}.npy")
    if not os.path.isfile(path):
        return {"error": f"数据集文件不存在: {path}"}
    a = np.load(path, mmap_mode="r")
    n = len(a)
    if idx is None:
        idx = int(np.random.randint(n))
    idx = max(0, min(n - 1, int(idx)))
    x = np.asarray(a[idx], np.float64)
    mean = float(x.mean())
    ac = x - mean
    ac_rms = float(np.sqrt((ac * ac).mean()))
    ds = x
    return {
        "class": cls,
        "class_name": SAMPLE_NAMES[cls],
        "class_cn": SAMPLE_CN[cls],
        "index": idx,
        "total": n,
        "fs": 10240,
        "step": 1,
        "ac_rms": ac_rms,
        "wave": [round(float(v), 5) for v in ds],
    }


def build_select_frame(cls: int, idx: int, seq: int):
    """把数据集某一窗转成板端可识别的 3×1024 int16 帧。"""
    import numpy as np
    cls = max(0, min(3, int(cls)))
    # 数据集文件是"按类分开"的；REPLAY_IDX 是全局拼接序号，要减去类偏移
    offsets = [0, 11165, 11165 + 6588, 11165 + 6588 + 57096]
    arrays = []
    local = None
    for ax in ("x", "y", "z"):
        p = os.path.join(DATA_DIR, f"trainData_{ax}_{cls}.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        a = np.load(p, mmap_mode="r")
        local = max(0, min(len(a) - 1, int(idx) - offsets[cls]))
        x = np.asarray(a[local], np.float64)
        codes = np.clip(np.round(2048.0 + x * 5.0 / 3.3 * 4095.0), 0, 4095).astype(np.int16)
        arrays.append(codes.tobytes())
    payload = b"".join(arrays)
    body = bytes([seq & 0xFF, cls & 0xFF]) + struct.pack("<I", int(idx)) + payload
    crc = 0
    for b in body:
        crc ^= b
    frame = b"\xA5\x5A" + body + bytes([crc & 0xFF])
    return frame, SAMPLE_NAMES[cls], SAMPLE_CN[cls]

RE_WINDOW = re.compile(r"--- window: replay class (\d+) \((\w+)\)(?: idx=(\d+))? ---")
RE_ACQ = re.compile(r"\[acq\s*\] (\d+) cycles \(([\d.]+) ms\).*adc_timeouts=(\d+)")
RE_WAVE = re.compile(r"\[wave\]\s+(.+)")
RE_ADC = re.compile(r"\[adc\s*\] mean=(-?\d+) rms_ac=([\d.]+)code \(([\d.]+)V, ([\d.]+) unit\) -> (idling|wear.*)")
RE_ADC_PRED = re.compile(r"wear, pred=(\d+) \((\w+)\)")
RE_REPLAY = re.compile(r"\[replay\] mean=(-?\d+) rms_ac=([\d.]+)code \(([\d.]+)V, ([\d.]+) unit\) -> (idling|wear.*)")
RE_REPLAY_PRED = re.compile(r"wear, pred=(\d+) \((\w+)\)")
RE_INFER = re.compile(r"infer=(?:(\d+) cycles \()?([\d.]+) ms\)?")
RE_PROB = re.compile(r"class (\d+) \((\w+)\): ([\d.]+)")
RE_WINS = re.compile(r"\[window\] acq ([\d.]+) ms \+ infer ([\d.]+) ms = ([\d.]+) ms/decision")
RE_SEL = re.compile(r"\[sel\] seq=(\d+) cls=(\d+) idx=(\d+) gate=(\w+) pred=(-?\d+) conf=([\d.]+)(?: rms=([\d.]+))?(?: p0=([\d.]+) p1=([\d.]+) p2=([\d.]+))?")


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.port = ""
        self.source = "未连接"
        self.frames = 0
        self.cur = None
        self.last = {}
        self.history = []
        self.ser = None
        self.sel_seq = 0
        self.sel_result = {}
        self.auto_enabled = False
        self.auto_stats = {"tested": 0, "correct": 0}
        self.auto_last = {}
        self.auto_history = []
        self.auto_lock = threading.Lock()
        self.model = "cnn"
        self.switching = False
        self.model_lock_until = 0.0

    def start_frame(self, cls_idx, cls_name, replay_idx=None):
        with self.lock:
            self.cur = {"t": time.strftime("%H:%M:%S"), "replay_class": cls_name,
                        "replay_idx": replay_idx if replay_idx is not None else cls_idx,
                        "replay_win": replay_idx,
                        "probs": [], "adc": {}, "replay": {},
                        "acq_ms": 0.0, "infer_ms": 0.0, "total_ms": 0.0, "wave": []}

    def finalize(self):
        with self.lock:
            if not self.cur:
                return
            f = self.cur
            self.cur = None
            self.frames += 1
            self.last = f
            self.history = ([f] + self.history)[:40]

    def next_seq(self):
        with self.lock:
            self.sel_seq = (self.sel_seq + 1) & 0xFF
            return self.sel_seq

    def send_bytes(self, data):
        with self.lock:
            if not self.connected or self.ser is None:
                raise RuntimeError("串口未连接")
            self.ser.write(data)
            self.ser.flush()

    def snapshot(self):
        with self.lock:
            return {
                "connected": self.connected,
                "port": self.port,
                "source": self.source,
                "frames": self.frames,
                "last": self.last,
                "history": self.history,
                "classes": CLASSES,
                "class_cn": {k: v for k, v in CLASS_CN.items() if k in CLASSES},
                "time": time.strftime("%H:%M:%S"),
                "model": self.model,
                "switching": self.switching,
                "auto": {
                    "enabled": self.auto_enabled,
                    "tested": self.auto_stats["tested"],
                    "correct": self.auto_stats["correct"],
                    "last": self.auto_last,
                    "history": self.auto_history,
                },
            }


STATE = State()


def parse_line(line: str):
    line = line.strip()
    low = line.lower()
    now = time.time()
    with STATE.lock:
        locked = now < STATE.model_lock_until
    if not locked:
        if "tiny_transformer" in low or ("transformer" in low and "model" in low):
            with STATE.lock:
                STATE.model = "transformer"
        elif "vib_cnn" in low:
            with STATE.lock:
                STATE.model = "cnn"
    m = RE_SEL.search(line)
    if m:
        pred = int(m.group(5))
        probs = []
        if m.group(8) is not None:
            probs = [float(m.group(8)), float(m.group(9)), float(m.group(10))]
        with STATE.lock:
            STATE.sel_result = {
                "seq": int(m.group(1)),
                "cls": int(m.group(2)),
                "idx": int(m.group(3)),
                "gate": m.group(4),
                "pred": pred,
                "pred_name": CLASSES[pred] if 0 <= pred < len(CLASSES) else "",
                "conf": float(m.group(6)),
                "rms": float(m.group(7)) if m.group(7) else None,
                "probs": probs,
            }
        return
    m = RE_WINDOW.search(line)
    if m:
        STATE.start_frame(int(m.group(1)), m.group(2),
                          int(m.group(3)) if m.group(3) else None)
        return
    cur = STATE.cur
    if cur is None:
        return
    m = RE_ACQ.search(line)
    if m:
        cur["acq_cycles"] = int(m.group(1))
        cur["acq_ms"] = float(m.group(2))
        cur["adc_timeouts"] = int(m.group(3))
        return
    m = RE_WAVE.search(line)
    if m:
        try:
            cur["wave"] = [int(v) for v in m.group(1).split()]
        except ValueError:
            cur["wave"] = []
        return
    m = RE_ADC.search(line)
    if m:
        d = {"mean": int(m.group(1)), "rms_code": float(m.group(2)),
             "rms_v": float(m.group(3)), "rms_units": float(m.group(4)),
             "decision": "idling" if m.group(5) == "idling" else "wear"}
        p = RE_ADC_PRED.search(line)
        if p:
            d["pred"] = int(p.group(1))
            d["pred_name"] = p.group(2)
        cur["adc"] = d
        return
    m = RE_REPLAY.search(line)
    if m:
        d = {"mean": int(m.group(1)), "rms_code": float(m.group(2)),
             "rms_v": float(m.group(3)), "rms_units": float(m.group(4)),
             "decision": "idling" if m.group(5) == "idling" else "wear"}
        p = RE_REPLAY_PRED.search(line)
        if p:
            d["pred"] = int(p.group(1))
            d["pred_name"] = p.group(2)
        q = RE_INFER.search(line)
        if q:
            if q.group(1):
                d["infer_cycles"] = int(q.group(1))
            d["infer_ms"] = float(q.group(2))
            cur["infer_ms"] = d["infer_ms"]
        cur["replay"] = d
        return
    m = RE_PROB.search(line)
    if m:
        cur["probs"].append(float(m.group(3)))
        return
    m = RE_WINS.search(line)
    if m:
        cur["acq_ms"] = float(m.group(1))
        cur["infer_ms"] = float(m.group(2))
        cur["total_ms"] = float(m.group(3))
        STATE.finalize()


def serial_worker(port: str | None, state: State):
    try:
        import serial  # pyserial
        from serial.tools import list_ports
    except ImportError:
        print("[serial] 缺少 pyserial，请: pip install pyserial", flush=True)
        return
    while True:
        try:
            p = port
            if not p:
                for info in list_ports.comports():
                    if "CH340" in (info.description or "") or "CH34" in (info.description or ""):
                        p = info.device
                        break
            if not p:
                state.connected = False
                state.source = "未找到串口（插好板子后自动重试）"
                time.sleep(2)
                continue
            state.port = p
            with serial.Serial(p, 115200, timeout=0.2) as ser:
                ser.dtr = False
                ser.rts = False
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                state.ser = ser
                state.connected = True
                state.source = "串口在线"
                print(f"[serial] {p} 已打开", flush=True)
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    parse_line(line)
        except Exception as e:  # 掉线自动重连
            state.connected = False
            state.ser = None
            state.source = f"串口异常: {type(e).__name__}"
            time.sleep(2)


def simulate_worker(state: State):
    import random
    state.connected = True
    state.port = "SIM"
    state.source = "模拟模式（未连接板子）"
    while True:
        cls = random.randrange(4)
        name = ["idling", "initial_wear", "steady_wear", "accelerating_wear"][cls]
        state.start_frame(cls, name)
        cur = state.cur
        acq = 100.0
        probs = [random.uniform(0.02, 0.08) for _ in range(3)]
        if cls > 0:
            probs[cls - 1] = random.uniform(0.85, 0.99)
            s = sum(probs)
            probs = [p / s for p in probs]
            cur["probs"] = probs
            cur["replay"] = {"decision": "wear", "pred": cls - 1, "pred_name": name,
                             "infer_ms": 17.07, "rms_v": random.uniform(0.06, 0.2),
                             "rms_units": random.uniform(0.01, 0.05)}
            cur["adc"] = dict(cur["replay"])
        else:
            cur["replay"] = {"decision": "idling", "rms_v": 0.006, "rms_units": 0.0016}
            cur["adc"] = {"decision": "idling", "rms_v": 0.001, "rms_units": 0.0002}
        # 模拟波形：和固件一样每 4 点取 1 点，256 点
        import math
        amp = 6.0 if cls == 0 else 60.0 + 40.0 * cls
        freq = 120.0 + 40.0 * cls
        cur["wave"] = [int(2048 + amp * math.sin(2 * math.pi * freq * i / 2560.0)
                           + random.uniform(-8, 8)) for i in range(256)]
        cur["acq_ms"] = acq
        cur["infer_ms"] = 17.07 if cls > 0 else 0.0
        cur["total_ms"] = acq + cur["infer_ms"]
        time.sleep(0.25)
        state.finalize()


def auto_worker(state: State):
    """自动从 90,221 个窗口里随机抽窗，发给板子测，并累计正确率。"""
    import random
    import numpy as np
    while True:
        if not state.auto_enabled:
            time.sleep(0.2)
            continue
        cls = random.randrange(4)
        local = random.randrange(SAMPLE_COUNTS[cls])
        idx = SAMPLE_OFFSETS[cls] + local
        try:
            with state.lock:
                state.sel_result = {}
            seq = state.next_seq()
            frame, name, cn = build_select_frame(cls, idx, seq)
            state.send_bytes(frame)
            result = None
            for _ in range(80):  # 最多等 4 s（板子主循环有 1.5 s 间隔）
                time.sleep(0.05)
                with state.lock:
                    r = dict(state.sel_result)
                if r.get("seq") == seq:
                    result = r
                    break
            if result is None:
                with state.lock:
                    state.auto_last = {"timeout": True, "true_cn": cn, "idx": idx}
                time.sleep(0.3)
                continue
            p = os.path.join(DATA_DIR, f"trainData_x_{cls}.npy")
            a = np.load(p, mmap_mode="r")
            x = np.asarray(a[local], np.float64)
            mean = float(x.mean())
            ac = x - mean
            ac_rms = float(np.sqrt((ac * ac).mean()))
            correct = ((cls == 0 and result["gate"] == "idling") or
                       (cls > 0 and result["gate"] == "wear" and result["pred"] == cls - 1))
            with state.lock:
                state.auto_stats["tested"] += 1
                if correct:
                    state.auto_stats["correct"] += 1
                state.auto_last = {
                    "true_class": name, "true_cn": cn, "idx": idx,
                    "ac_rms": ac_rms,
                    "wave": [round(float(v), 5) for v in x[::4]],
                    "board": result, "correct": correct,
                }
                state.auto_history.insert(0, {
                    "t": time.strftime("%H:%M:%S"),
                    "true_class": name, "true_cn": cn, "idx": idx,
                    "board": result, "correct": correct,
                })
                del state.auto_history[30:]
            time.sleep(0.15)
        except Exception as e:
            with state.lock:
                state.auto_last = {"error": str(e), "true_cn": cn, "idx": idx}
            time.sleep(1.0)


def flash_firmware(state: State, name: str):
    """通过 OpenOCD + DAP 重新烧录指定模型固件。"""
    info = FIRMWARES.get(name)
    if not info:
        raise RuntimeError(f"未知模型: {name}")
    elf = info["elf"]
    if not os.path.isfile(elf):
        raise RuntimeError(f"固件不存在: {elf}")
    cmd = [OPENOCD,
           "-f", "interface/cmsis-dap.cfg",
           "-f", "target/stm32f1x.cfg",
           "-c", "adapter speed 500",
           "-c", "init",
           "-c", "program " + elf.replace("\\", "/") + " verify reset",
           "-c", "reset run",
           "-c", "exit"]
    state.connected = False
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if "Verified OK" not in out:
        raise RuntimeError("烧录失败：" + out[-500:])
    time.sleep(2.5)
    # 烧录后板子复位，主动关掉旧串口句柄，让 serial_worker 重新打开 COM
    with state.lock:
        if state.ser is not None:
            try:
                state.ser.close()
            except Exception:
                pass
            state.ser = None
        state.connected = False
        state.model = name
        state.model_lock_until = time.time() + 10.0
    return info["label"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, content, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = os.path.join(ROOT, "demo", "static", "index.html")
            with open(p, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/model"):
            q = parse_qs(urlparse(self.path).query)
            name = q.get("name", ["cnn"])[0]
            if name not in FIRMWARES:
                body = json.dumps({"error": "未知模型"}, ensure_ascii=False).encode("utf-8")
                self._send(400, body, "application/json; charset=utf-8")
            elif STATE.switching:
                body = json.dumps({"error": "正在切换中"}, ensure_ascii=False).encode("utf-8")
                self._send(409, body, "application/json; charset=utf-8")
            else:
                STATE.switching = True
                STATE.auto_enabled = False
                try:
                    label = flash_firmware(STATE, name)
                    body = json.dumps({"ok": True, "model": name, "label": label},
                                      ensure_ascii=False).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                except Exception as e:
                    body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                    self._send(500, body, "application/json; charset=utf-8")
                finally:
                    STATE.switching = False
        elif self.path.startswith("/api/auto"):
            q = parse_qs(urlparse(self.path).query)
            on = q.get("on", ["0"])[0] in ("1", "true", "on")
            reset = q.get("reset", ["0"])[0] in ("1", "true")
            with STATE.lock:
                STATE.auto_enabled = on
                if reset:
                    STATE.auto_stats = {"tested": 0, "correct": 0}
                    STATE.auto_last = {}
            body = json.dumps({"enabled": on,
                               "tested": STATE.auto_stats["tested"],
                               "correct": STATE.auto_stats["correct"]},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif self.path.startswith("/api/select"):
            q = parse_qs(urlparse(self.path).query)
            try:
                cls = int(q.get("cls", ["0"])[0])
                idx = int(q.get("idx", ["0"])[0])
                with STATE.lock:
                    STATE.sel_result = {}
                seq = STATE.next_seq()
                frame, name, cn = build_select_frame(cls, idx, seq)
                STATE.send_bytes(frame)
                result = None
                for _ in range(50):  # 最多等 5 s
                    time.sleep(0.1)
                    with STATE.lock:
                        r = dict(STATE.sel_result)
                    if r.get("seq") == seq:
                        result = r
                        break
                payload = {
                    "true_class": name,
                    "true_cn": cn,
                    "idx": idx,
                    "seq": seq,
                    "board": result,
                    "timeout": result is None,
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
        elif self.path.startswith("/api/sample"):
            q = parse_qs(urlparse(self.path).query)
            try:
                cls = int(q.get("cls", ["0"])[0])
                idx = q.get("idx", [None])[0]
                idx = int(idx) if idx not in (None, "") else None
                body = json.dumps(sample_payload(cls, idx), ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
        elif self.path.startswith("/api/state"):
            body = json.dumps(STATE.snapshot(), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif self.path.startswith("/static/"):
            rel = os.path.basename(self.path[len("/static/"):].split("?")[0])
            p = os.path.join(ROOT, "demo", "static", rel)
            if os.path.isfile(p):
                ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
                ctype = {"html": "text/html; charset=utf-8",
                         "png": "image/png",
                         "css": "text/css; charset=utf-8",
                         "js": "application/javascript; charset=utf-8"}.get(ext, "application/octet-stream")
                with open(p, "rb") as f:
                    self._send(200, f.read(), ctype)
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="串口，如 COM6；不填自动找 CH340")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    t = threading.Thread(target=simulate_worker, args=(STATE,), daemon=True) if args.simulate \
        else threading.Thread(target=serial_worker, args=(args.port, STATE), daemon=True)
    t.start()
    threading.Thread(target=auto_worker, args=(STATE,), daemon=True).start()

    url = f"http://127.0.0.1:{args.http_port}"
    print(f"[http] {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
