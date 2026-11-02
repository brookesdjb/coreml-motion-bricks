"""Serve the crowd street: NPCs split over worker processes (each with its own Python interpreter and
Neural Engine batcher, as in cmb.crowd.stress), streamed to demos/web/crowd.html.

Each worker steps its NPCs at 15 Hz on a shared wall-clock schedule and publishes joint positions through
shared memory. The main process forwards every new worker frame to the browser as a binary message tagged
with the worker id and its simulation time; the page interpolates each worker's NPCs separately.

Run: python demos/crowd.py --npcs 250 --procs 6   then open http://localhost:8775
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import shared_memory
from pathlib import Path
from threading import Thread

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

WEB = Path(__file__).parent / "web"

class _Handler(SimpleHTTPRequestHandler):
    """Static files from demos/web; "/" opens this demo's page."""
    page = "crowd.html"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            self.path = "/" + self.page
        return super().do_GET()

    def log_message(self, *args):
        pass

# The model's 30 joints are enough to draw a body; skip fingers.
DRAW_JOINTS = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
               "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm",
               "RightHand", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot",
               "RightToeBase"]
J = len(DRAW_JOINTS)
# Shared header per worker (float64): seq (odd while writing), sim_time, tick_ms, ne_ms, gens, interacting, ready.
HEADER = 8


def draw_parents(names, parents_full):
    idx = [names.index(j) for j in DRAW_JOINTS]
    parents = []
    for j in idx:
        p = parents_full[j]
        while p >= 0 and p not in idx:  # nearest drawn ancestor
            p = parents_full[p]
        parents.append(idx.index(p) if p >= 0 else -1)
    return idx, parents


def worker(wid, n, hz, header_name, joints_name, start_at):
    """Run n NPCs and publish their poses after every tick."""
    from cmb.crowd import Crowd
    time.sleep(wid * 5)  # small stagger so workers don't all build Warp kernels at once
    crowd = Crowd(n, seed=wid)
    s0 = crowd.npcs[0].s
    idx, _ = draw_parents(s0.joint_names, s0.parents)
    hdr_shm = shared_memory.SharedMemory(name=header_name)
    jnt_shm = shared_memory.SharedMemory(name=joints_name)
    hdr = np.ndarray((HEADER,), dtype=np.float64, buffer=hdr_shm.buf)
    jnt = np.ndarray((n, J, 3), dtype=np.float32, buffer=jnt_shm.buf)
    dt = 1 / hz
    b = crowd.batcher
    hdr[6] = 1.0  # ready
    import os
    parent = os.getppid()
    while time.time() < start_at:
        time.sleep(0.01)
    k = 0
    while True:
        if os.getppid() != parent:  # server was killed without cleaning up: don't linger as an orphan
            return
        busy0, gens0 = b.busy_s, b.generations
        t0 = time.perf_counter()
        crowd.tick(dt)
        tick_ms = (time.perf_counter() - t0) * 1000
        k += 1
        joints = crowd.joints()[:, idx, :]
        states = [type(npc.s.machine.state).__name__ for npc in crowd.npcs]
        hdr[0] += 1  # odd: writing
        jnt[:] = joints
        hdr[1:6] = (k * dt, tick_ms, (b.busy_s - busy0) * 1000, b.generations - gens0,
                    states.count("InteractionState"))
        hdr[0] += 1  # even: consistent
        # Keep to the shared schedule; when behind, carry on without sleeping (the page shows the lag).
        wait = start_at + k * dt - time.time()
        if wait > 0:
            time.sleep(wait)


class CrowdServer:
    def __init__(self, n, procs, hz):
        from cmb.sim import SharedAssets
        from cmb.street import street_scene
        self.hz, self.procs = hz, procs
        self.counts = [n // procs + (i < n % procs) for i in range(procs)]
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(int).tolist()
        shared = SharedAssets(street_scene())  # main process only needs the scene and skeleton
        self.scene = shared.scene
        sk = shared.skeleton
        names = [sk.joint_name(i) for i in range(sk.num_joints)]
        _, self.parents = draw_parents(names, sk.parent_indices.numpy().tolist())
        self.hdr_shm = [shared_memory.SharedMemory(create=True, size=HEADER * 8) for _ in range(procs)]
        self.jnt_shm = [shared_memory.SharedMemory(create=True, size=max(1, c) * J * 3 * 4) for c in self.counts]
        self.hdrs = [np.ndarray((HEADER,), dtype=np.float64, buffer=s.buf) for s in self.hdr_shm]
        self.jnts = [np.ndarray((c, J, 3), dtype=np.float32, buffer=s.buf) for c, s in zip(self.counts, self.jnt_shm)]
        for h in self.hdrs:
            h[:] = 0
        self.start_at = time.time() + 5 * procs + 30 + 1.5 * max(self.counts)
        ctx = mp.get_context("spawn")
        self.workers = [ctx.Process(target=worker, daemon=True,
                                    args=(i, c, hz, self.hdr_shm[i].name, self.jnt_shm[i].name, self.start_at))
                        for i, c in enumerate(self.counts)]
        for w in self.workers:
            w.start()
        self.clients = set()
        self.last_seq = [0] * procs

    def scene_msg(self):
        cubes = [{"p": list(map(float, c.transform.p)), "q": list(map(float, c.transform.q)),
                  "size": [float(c.width), float(c.depth), float(c.height)], "color": list(map(float, c.color))}
                 for c in self.scene.solid_cubes]
        return json.dumps({"type": "scene", "cubes": cubes, "parents": self.parents, "joints": DRAW_JOINTS,
                           "npcs": sum(self.counts), "hz": self.hz, "procs": self.procs})

    def read_worker(self, i):
        """Return (header copy, joints copy) if worker i has a new consistent frame, else None."""
        h = self.hdrs[i]
        for _ in range(3):
            seq = h[0]
            if seq % 2 or seq == self.last_seq[i]:
                return None if seq == self.last_seq[i] else None
            hc, jc = h.copy(), self.jnts[i].copy()
            if h[0] == seq:
                self.last_seq[i] = seq
                return hc, jc
        return None

    async def loop(self):
        print(f"starting {self.procs} workers for {sum(self.counts)} NPCs; streaming begins in "
              f"~{self.start_at - time.time():.0f}s", flush=True)
        stats = [dict(tick_ms=0, ne_ms=0, gens=0, interacting=0, lag=0) for _ in range(self.procs)]
        last_stats = time.time()
        while True:
            for i in range(self.procs):
                got = self.read_worker(i)
                if got is None:
                    continue
                hc, jc = got
                stats[i].update(tick_ms=hc[2], ne_ms=hc[3], interacting=int(hc[5]),
                                lag=time.time() - self.start_at - hc[1])
                stats[i]["gens"] += int(hc[4])
                if self.clients:
                    head = np.array([i, hc[1], self.starts[i], self.counts[i]], dtype=np.float32)
                    websockets.broadcast(self.clients, np.concatenate([head, jc.ravel()]).tobytes())
            if time.time() - last_stats > 0.5 and self.clients:
                el = time.time() - last_stats
                last_stats = time.time()
                msg = {"type": "stats", "budget_ms": round(1000 / self.hz),
                       "tick_ms": round(max(s["tick_ms"] for s in stats), 1),
                       "ne_ms": round(sum(s["ne_ms"] for s in stats), 1),
                       "gens_per_s": round(sum(s["gens"] for s in stats) / el),
                       "interacting": sum(s["interacting"] for s in stats),
                       "lag_s": round(max(s["lag"] for s in stats), 2),
                       "ready": int(sum(h[6] for h in self.hdrs)), "procs": self.procs}
                for s in stats:
                    s["gens"] = 0
                websockets.broadcast(self.clients, json.dumps(msg))
            await asyncio.sleep(0.004)

    async def handle(self, ws):
        self.clients.add(ws)
        await ws.send(self.scene_msg())
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    def close(self):
        for w in self.workers:
            w.terminate()
        for s in self.hdr_shm + self.jnt_shm:
            s.close()
            s.unlink()


async def main(args):
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    server = CrowdServer(args.npcs, args.procs, args.hz)
    try:
        async with websockets.serve(server.handle, "127.0.0.1", args.port + 1, max_size=None):
            print(f"ready: http://localhost:{args.port}", flush=True)
            await server.loop()
    finally:
        server.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8775)
    ap.add_argument("--npcs", type=int, default=250)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--hz", type=float, default=15.0)
    asyncio.run(main(ap.parse_args()))
