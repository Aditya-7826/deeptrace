"""Measure the API's peak RSS under load and check it stays under the 1.5 GB budget.

PowerShell / bash:
    uv run python scripts/measure_api_memory.py --weights-dir weights

Starts `deeptrace serve` in a subprocess, samples its RSS every 0.1 s while sending
sequential and concurrent /predict requests (each image re-encoded so the cache never
answers), then /explain requests (which load the FP32 Grad-CAM model). Writes
results/api_memory.json and exits 1 if the peak exceeds --limit-mb.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ["real/aldrin.jpg", "real/multiface.jpg", "real/jemison_rgba.png", "fake/ochoa.jpg",
          "fake/whitson.jpg", "real/noface.jpg"]  # fmt: skip


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def variant(path: Path, k: int) -> bytes:
    """Distinct bytes per request (re-encoded JPEG) so the LRU cache never short-circuits."""
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=80 + k % 15)
        return buf.getvalue()


def post(url: str, data: bytes) -> int:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})  # fmt: skip
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


class Sampler(threading.Thread):
    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.proc = psutil.Process(pid)
        self.peak = 0
        self.current = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                rss += sum(c.memory_info().rss for c in self.proc.children(recursive=True))
                self.current, self.peak = rss, max(self.peak, rss)
            except psutil.Error:
                pass
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--requests", type=int, default=40, help="per phase")
    ap.add_argument("--explains", type=int, default=4)
    ap.add_argument("--limit-mb", type=float, default=1500.0)
    ap.add_argument("--out", default="results/api_memory.json")
    args = ap.parse_args(argv)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "deeptrace", "serve", "--weights-dir", args.weights_dir,
           "--port", str(port)]  # fmt: skip
    # The rate limit would otherwise answer later requests with 429 and hide their memory use.
    env = {**os.environ, "DEEPTRACE_RATE_LIMIT": "1000000"}
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    sampler = Sampler(server.pid)
    sampler.start()
    phases: dict[str, dict] = {}
    try:
        t0 = time.time()
        while True:
            if server.poll() is not None:
                raise SystemExit(f"server exited: {server.stderr.read().decode(errors='replace')}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2):
                    break
            except OSError:
                if time.time() - t0 > 180:
                    raise SystemExit("server did not become healthy within 180 s") from None
                time.sleep(0.5)
        time.sleep(1.0)
        phases["startup"] = {"rss_mb": round(sampler.current / 1e6, 1),
                             "seconds": round(time.time() - t0, 1)}  # fmt: skip
        files = [ROOT / "tests" / "fixtures" / "images" / p for p in IMAGES]

        statuses = [post(base + "/predict", variant(files[k % len(files)], k))
                    for k in range(args.requests)]  # fmt: skip
        phases["sequential_predict"] = {"n": len(statuses), "statuses": sorted(set(statuses)),
                                        "rss_mb": round(sampler.current / 1e6, 1)}  # fmt: skip
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(
                lambda k: post(base + "/predict", variant(files[k % len(files)], 1000 + k)),
                range(args.requests)))  # fmt: skip
        phases["concurrent_predict_4_clients"] = {
            "n": len(statuses), "statuses": sorted(set(statuses)),
            "rss_mb": round(sampler.current / 1e6, 1)}  # fmt: skip
        statuses = [post(base + "/explain?face_index=0", variant(files[0], 2000 + k))
                    for k in range(args.explains)]  # fmt: skip
        phases["explain_loads_fp32_gradcam"] = {
            "n": len(statuses),
            "statuses": sorted(set(statuses)),
            "rss_mb": round(sampler.current / 1e6, 1),
        }
        time.sleep(0.5)
    finally:
        sampler.stop.set()
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

    meta = json.loads((Path(args.weights_dir) / "model.json").read_text(encoding="utf-8"))
    from deeptrace.export.benchmark import system_info

    peak_mb = round(sampler.peak / 1e6, 1)
    # Only valid if every request was really served (no 429/5xx short-circuits).
    all_ok = all(p.get("statuses") == [200] for k, p in phases.items() if k != "startup")
    report = {
        "measured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": system_info(),
        "model": {"version": meta["version"], "serving_model": meta["serving_model"]},
        "server": "deeptrace serve (1 uvicorn worker, 2 inference slots, 2 threads; "
        "rate limit lifted for the measurement)",
        "phases": phases,
        "peak_rss_mb": peak_mb,
        "limit_mb": args.limit_mb,
        "all_requests_ok": all_ok,
        "passed": all_ok and peak_mb < args.limit_mb,
    }  # fmt: skip
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"peak_rss_mb": peak_mb, "limit_mb": args.limit_mb,
                      "all_requests_ok": all_ok, "passed": report["passed"],
                      "phases": phases}, indent=2))  # fmt: skip
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
