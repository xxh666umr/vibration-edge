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
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in (os.path.join(ROOT, ".deps"), os.environ.get("EDGE_DEPS", "")):
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

CLASSES = ["initial_wear", "steady_wear", "accelerating_wear"]
CLASS_CN = {"idling": "空闲", "initial_wear": "初期磨损",
            "steady_wear": "稳定磨损", "accelerating_wear": "加速磨损"}

RE_WINDOW = re.compile(r"--- window: replay class (\d+) \((\w+)\) ---")
RE_ACQ = re.compile(r"\[acq\s*\] (\d+) cycles \(([\d.]+) ms\).*adc_timeouts=(\d+)")
RE_ADC = re.compile(r"\[adc\s*\] mean=(-?\d+) rms_ac=([\d.]+)code \(([\d.]+)V, ([\d.]+) unit\) -> (idling|wear.*)")
RE_ADC_PRED = re.compile(r"wear, pred=(\d+) \((\w+)\)")
RE_REPLAY = re.compile(r"\[replay\] mean=(-?\d+) rms_ac=([\d.]+)code \(([\d.]+)V, ([\d.]+) unit\) -> (idling|wear.*)")
RE_REPLAY_PRED = re.compile(r"wear, pred=(\d+) \((\w+)\)")
RE_INFER = re.compile(r"infer=(?:(\d+) cycles \()?([\d.]+) ms\)?")
RE_PROB = re.compile(r"class (\d+) \((\w+)\): ([\d.]+)")
RE_WINS = re.compile(r"\[window\] acq ([\d.]+) ms \+ infer ([\d.]+) ms = ([\d.]+) ms/decision")


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

    def start_frame(self, cls_idx, cls_name):
        with self.lock:
            self.cur = {"t": time.strftime("%H:%M:%S"), "replay_class": cls_name,
                        "replay_idx": cls_idx, "probs": [], "adc": {}, "replay": {},
                        "acq_ms": 0.0, "infer_ms": 0.0, "total_ms": 0.0}

    def finalize(self):
        with self.lock:
            if not self.cur:
                return
            f = self.cur
            self.cur = None
            self.frames += 1
            self.last = f
            self.history = ([f] + self.history)[:40]

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
            }


STATE = State()


def parse_line(line: str):
    line = line.strip()
    m = RE_WINDOW.search(line)
    if m:
        STATE.start_frame(int(m.group(1)), m.group(2))
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
        cur["acq_ms"] = acq
        cur["infer_ms"] = 17.07 if cls > 0 else 0.0
        cur["total_ms"] = acq + cur["infer_ms"]
        time.sleep(0.25)
        state.finalize()


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
