"""Serve the browser obstacle course: static files over HTTP, and the MotionBricks character over a WebSocket.

The browser sends its input state; the server steps the character at 60 Hz and sends back world joint positions.
Run: python demos/course.py   then open http://localhost:8765
"""
import argparse
import asyncio
import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from cmb.sim import RESPONSIVE, Session  # noqa: E402
from app.runtime.common.motion import CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE  # noqa: E402

WEB = Path(__file__).parent / "web"

class _Handler(SimpleHTTPRequestHandler):
    """Static files from demos/web; "/" opens this demo's page."""
    page = "course.html"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            self.path = "/" + self.page
        return super().do_GET()

    def log_message(self, *args):
        pass

TICK = 1 / 60


class Game:
    def __init__(self, backend, responsive=False):
        self.session = Session(backend=backend, settings=RESPONSIVE if responsive else None)
        self.input = {"move": [0, 0], "facing": [0, 1], "run": False, "interact": False}
        self.pending_mode = None
        self.clients = set()
        # Every open page sends input 30x a second; only the page the player last pressed a key in drives.
        self.controller = None
        self.stats = {"step_ms": 0.0, "gen_ms": 0.0}
        self.sim_time = 0.0

    def toggle(self, kind):
        machine = self.session.machine
        current = machine.active_locomotion_state.active_mode.name
        styles = tuple(getattr(machine, "style_modes", ()))
        if kind == "crouch":
            self.pending_mode = LINEAR_LOCOMOTION_MODE if current == CROUCH_LOCOMOTION_MODE else CROUCH_LOCOMOTION_MODE
        elif kind == "strafe":
            self.pending_mode = LINEAR_LOCOMOTION_MODE if current == DIRECTIONAL_LOCOMOTION_MODE else DIRECTIONAL_LOCOMOTION_MODE
        elif kind == "style" and styles:
            self.pending_mode = styles[(styles.index(current) + 1) % len(styles)] if current in styles else styles[0]
        elif kind == "normal":
            self.pending_mode = LINEAR_LOCOMOTION_MODE

    def frame(self):
        s = self.session
        return {
            "type": "frame",
            "t": round(self.sim_time, 4),
            "joints": [round(float(v), 4) for v in s.world_joints().ravel()],
            "state": type(s.machine.state).__name__.replace("State", ""),
            "mode": s.machine.active_locomotion_state.active_mode.name,
            "near": [p["def"].name for p in s.primitives if p.get("hit")],
            "step_ms": round(self.stats["step_ms"], 1),
            "gen_ms": round(self.session.engine.timer.milliseconds.get("inference", 0.0), 1),
            "backend": self.session.engine.backend,
        }

    async def loop(self):
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt, last = now - last, now
            i = self.input
            t0 = time.perf_counter()
            self.session.step(dt, move=i["move"], facing=i["facing"], run=i["run"],
                              mode=self.pending_mode, interact=i.pop("interact_pressed", False))
            self.pending_mode = None
            self.sim_time += min(max(dt, 1e-6), 0.1)  # same clamp as Session.step
            self.stats["step_ms"] = (time.perf_counter() - t0) * 1000
            if self.clients:
                msg = json.dumps(self.frame())
                websockets.broadcast(self.clients, msg)
            await asyncio.sleep(max(0.0, TICK - (time.perf_counter() - now)))

    async def handle(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({"type": "scene", **self.session.scene_description()}))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("active") or msg.get("type") in ("mode", "auto"):
                    self.controller = ws
                if self.controller is not None and ws is not self.controller:
                    continue
                if msg.get("type") == "input":
                    self.input.update(move=msg["move"], facing=msg["facing"], run=bool(msg["run"]))
                    if msg.get("interact"):
                        self.input["interact_pressed"] = True
                elif msg.get("type") == "mode":
                    self.toggle(msg["kind"])
                elif msg.get("type") == "auto":
                    self.session.auto_interact = bool(msg["on"])
        finally:
            self.clients.discard(ws)
            if ws is self.controller:
                self.controller = None
                self.input.update(move=[0, 0], run=False)


async def main(args):
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    game = Game(args.backend, responsive=args.responsive)
    async with websockets.serve(game.handle, "127.0.0.1", args.port + 1):
        print(f"ready: http://localhost:{args.port}  (backend {args.backend})", flush=True)
        await game.loop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    # CPU_AND_NE runs the network on the Neural Engine and keeps the GPU free.
    ap.add_argument("--responsive", action="store_true",
                    help="faster input smoothing than NVIDIA's defaults (cmb.sim.RESPONSIVE); not part of MotionBricks")
    ap.add_argument("--backend", default="CPU_AND_NE", help="ort, CPU_ONLY, CPU_AND_NE, CPU_AND_GPU or ALL")
    asyncio.run(main(ap.parse_args()))
