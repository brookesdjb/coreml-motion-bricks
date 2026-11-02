"""A crowd of MotionBricks NPCs on a street, with generations batched onto the Neural Engine.

Each NPC is a full NVIDIA SOMA character (cmb.sim.Session). Within a process, NPCs step on their own threads;
when one needs a new clip it blocks in Batcher.request. Once every NPC still running this tick is finished or
waiting, the batcher sends all waiting requests to CoreML together (batch-8/4/2/1 models, several calls queued
in one batch-API submission) and hands back the results. For more NPCs, split them over worker processes.

Stress test (headless):
    python -m cmb.crowd 100 200 300 --procs 6        # 15 Hz, calm crowd profile
    python -m cmb.crowd 50 --procs 1 --nvidia-settings

On an M2 Pro: ~300 NPCs in real time with 6 processes and the calm profile (see README).
"""
import argparse
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from .engine import MacSOMAEngine, load_coreml, split_outputs
from .paths import LOG_DIR, coreml_package
from .sim import Session, SharedAssets
from .street import LANE_HALF_WIDTH, STREET_END, STREET_START, street_scene

import warp as wp  # noqa: E402  (after cmb.engine, which sets up the import shims)

BATCH_SIZES = (8, 4, 2, 1)


def split_sizes(n, sizes=BATCH_SIZES):
    """Batch sizes covering n requests: the largest model at least 3/4 full, padding the rest."""
    out = []
    while n > 0:
        size = next((b for b in sorted(sizes, reverse=True) if b <= n or n >= 0.75 * b), 1)
        out.append(size)
        n -= min(size, n)
    return out


def run_requests(models, feeds_list, keepalive):
    """Group requests into batch sizes, submit each size through the batch API, split the results."""
    names = list(feeds_list[0])
    groups, i = [], 0
    for size in split_sizes(len(feeds_list), models.keys()):
        groups.append((size, i, feeds_list[i:i + size]))
        i += size
    results = [None] * len(feeds_list)
    for size in sorted({g[0] for g in groups}, reverse=True):
        same = [g for g in groups if g[0] == size]
        chunks = []
        for _, _, part in same:
            part = list(part) + [part[-1]] * (size - len(part))  # pad
            chunks.append({n: np.concatenate([f[n] for f in part]) for n in names})
        keepalive.append(chunks)  # CoreML must not drop the last reference (see Batcher)
        outs = models[size].predict(chunks) if len(chunks) > 1 else [models[size].predict(chunks[0])]
        for (_, offset, part), out in zip(same, outs):
            motion, tokens = split_outputs(out, size)
            for j in range(len(part)):
                results[offset + j] = (motion[j][None].copy(), tokens[j:j + 1].copy())
    return results


class Batcher:
    """Collects generation requests from NPC threads and runs them together on CoreML."""

    def __init__(self, units="CPU_AND_NE"):
        self.models = {b: load_coreml(coreml_package(b), units) for b in BATCH_SIZES}
        self.cond = threading.Condition()
        self.pending = []  # (feeds, result slot)
        self.running = 0
        self.busy_s = 0.0
        self.generations = 0
        # CoreML frees its inputs later on its own queue without the GIL; if that drops the last reference to a
        # numpy input, Python segfaults (MLE5ExecutionStream.resetQueue -> _PyObject_Free). Holding recent
        # inputs here makes Python free them instead.
        self._keepalive = deque(maxlen=512)

    def begin_tick(self, n):
        with self.cond:
            self.running = n

    def end_step(self):
        with self.cond:
            self.running -= 1
            self._maybe_flush()

    def request(self, feeds):
        slot = {}
        with self.cond:
            self.pending.append((feeds, slot))
            self._maybe_flush()
            while "motion" not in slot and "error" not in slot:
                self.cond.wait()
        if "error" in slot:
            raise RuntimeError("batched generation failed") from slot["error"]
        return slot["motion"], slot["tokens"]

    def _maybe_flush(self):
        # Called with the lock held: flush once every still-running NPC is blocked in request().
        if not self.pending or len(self.pending) < self.running:
            return
        batch, self.pending = self.pending, []
        t0 = time.perf_counter()
        try:
            try:
                results = run_requests(self.models, [f for f, _ in batch], self._keepalive)
            except Exception as first:  # e.g. a transient Neural Engine error with several processes sharing it
                print(f"[batcher] prediction failed, retrying once: {first!r}", file=sys.stderr, flush=True)
                results = run_requests(self.models, [f for f, _ in batch], self._keepalive)
        except Exception as exc:
            # Every waiting NPC must hear about it, or they wait forever.
            print(f"[batcher] prediction failed twice: {exc!r}", file=sys.stderr, flush=True)
            for _, slot in batch:
                slot["error"] = exc
            self.cond.notify_all()
            return
        self.busy_s += time.perf_counter() - t0
        self.generations += len(batch)
        for (_, slot), (motion, tokens) in zip(batch, results):
            slot["motion"], slot["tokens"] = motion, tokens
        self.cond.notify_all()


def apply_calm_profile(session):
    """Crowd settings that ask the model for ~40% fewer clips at the same motion quality: longer clips (12-16
    tokens), targets further ahead, and regeneration only on clearly different direction (~40 deg) or speed."""
    from app.runtime.common.motion import LocomotionSettings
    session.set_settings(LocomotionSettings(
        facing_halflife=0.15, velocity_direction_halflife=0.25, speed_acceleration_halflife=0.6,
        speed_deceleration_halflife=0.4, speed_regeneration_fraction=0.5, minimum_speed_regeneration_fraction=0.2,
        direction_regeneration_dot=0.75))
    loc = session.machine.active_locomotion_state
    for name, rt in list(loc.mode_runtimes.items()):  # frozen dataclasses: swap in copies
        d = rt.definition
        new = replace(rt, definition=replace(d, allowed_token_counts=tuple(range(12, 17)),
                                             position_scale=d.position_scale * 1.5,
                                             transition_early_by_frames=min(d.transition_early_by_frames, 6)))
        loc.mode_runtimes[name] = new
        if loc.active_mode_runtime is rt:
            loc.active_mode_runtime = new


class NPC:
    """Walks up and down the street in a lane, turning round at each end."""

    def __init__(self, session, rng, calm=True):
        self.s = session
        self.calm = calm
        self.heading = None
        self.lane = rng.uniform(-LANE_HALF_WIDTH, LANE_HALF_WIDTH)
        self.dir = 1.0 if rng.random() < 0.5 else -1.0
        self.run = rng.random() < 0.2
        self.wait = rng.uniform(0.0, 3.0)  # staggered starts spread generations across ticks

    def step(self, dt):
        if self.wait > 0:
            self.wait -= dt
            self.s.step(dt)  # idle
            return
        x, y = self.s.character.character_ground_position[:2]
        if y > STREET_END:
            self.dir = -1.0
        elif y < STREET_START:
            self.dir = 1.0
        if self.calm:
            # Hold a heading; only re-aim at the lane when drifting off it or at the street ends.
            off = self.lane - x
            if self.heading is None or self.heading[1] * self.dir <= 0 or abs(off) > 1.0:
                self.heading = np.array([np.clip(off * 0.15, -0.3, 0.3), self.dir])
                self.heading /= np.linalg.norm(self.heading)
            elif abs(off) < 0.2 and self.heading[0] != 0.0:
                self.heading = np.array([0.0, self.dir])
            move = self.heading
        else:
            move = np.array([(self.lane - x) * 0.4, self.dir])
            move /= np.linalg.norm(move)
        self.s.step(dt, move=tuple(move), facing=tuple(move), run=self.run)


class Crowd:
    def __init__(self, n, seed=0, units="CPU_AND_NE", calm=True):
        rng = np.random.default_rng(seed)
        self.shared = SharedAssets(street_scene())
        self.batcher = Batcher(units)
        self.npcs = []
        for _ in range(n):
            y = rng.uniform(STREET_START + 1, STREET_END - 1)
            x = rng.uniform(-LANE_HALF_WIDTH, LANE_HALF_WIDTH)
            start = wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity())
            engine = MacSOMAEngine(self.batcher.request, model_to_runtime=self.shared.to_runtime)
            session = Session(shared=self.shared, engine=engine, start=start)  # NVIDIA's settings unless calm
            if calm:
                apply_calm_profile(session)
            self.npcs.append(NPC(session, rng, calm=calm))
        self.pool = ThreadPoolExecutor(max_workers=max(1, n))
        self.errors = 0

    def tick(self, dt):
        self.batcher.begin_tick(len(self.npcs))

        def run(npc):
            try:
                npc.step(dt)
            finally:
                self.batcher.end_step()
        errors = [e for e in (f.exception() for f in [self.pool.submit(run, npc) for npc in self.npcs]) if e]
        if errors:
            self.errors += len(errors)
            print(f"[crowd] {len(errors)} NPC steps failed this tick: {errors[0]!r}", file=sys.stderr, flush=True)

    def joints(self):
        return np.stack([npc.s.world_joints() for npc in self.npcs])  # stack copies the reused buffers


# --- stress test ----------------------------------------------------------------------------------------------

def _measure(crowd, dt, seconds):
    for _ in range(int(3 / dt)):  # warm up past the staggered starts
        crowd.tick(dt)
    b = crowd.batcher
    b.busy_s = b.generations = 0
    t0 = time.perf_counter()
    for _ in range(int(seconds / dt)):
        crowd.tick(dt)
    return {"wall": time.perf_counter() - t0, "gens": b.generations, "busy": b.busy_s, "errors": crowd.errors}


def _worker(args):
    n, seed, hz, seconds, start_at, calm, out = args
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"crowd_worker{seed}.log", "w", buffering=1)
    import faulthandler
    faulthandler.dump_traceback_later(max(60.0, start_at - time.time() + 4 * seconds + 60), exit=True, file=log)
    time.sleep(seed * 5)  # stagger so workers don't all build Warp kernels at once
    crowd = Crowd(n, seed=seed, calm=calm)
    log.write("loaded\n")
    dt = 1 / hz
    for _ in range(int(3 / dt)):
        crowd.tick(dt)
    while time.time() < start_at:  # every worker measures over the same window
        time.sleep(0.01)
    out.put(_measure(crowd, dt, seconds))
    log.write("done\n")


def stress(n, procs=6, hz=15.0, seconds=12.0, calm=True):
    """Run n NPCs split over `procs` worker processes and report the real-time factor of the slowest."""
    if procs <= 1:
        res = [_measure(Crowd(n, calm=calm), 1 / hz, seconds)]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        per = [n // procs + (i < n % procs) for i in range(procs)]
        start_at = time.time() + 5 * procs + 40 + 1.5 * max(per)
        out = ctx.Queue()
        workers = [ctx.Process(target=_worker, args=((k, i, hz, seconds, start_at, calm, out),))
                   for i, k in enumerate(per)]
        for w in workers:
            w.start()
        res = [out.get() for _ in workers]
        for w in workers:
            w.join()
    wall = max(r["wall"] for r in res)
    print(f"{n:>4} NPCs, {procs} process{'es' if procs > 1 else ''}, {hz:.0f} Hz, "
          f"{'calm crowd' if calm else 'NVIDIA default'} settings | real-time x{seconds / wall:4.2f} | "
          f"{sum(r['gens'] for r in res) / seconds:4.0f} clips/s ({sum(r['gens'] for r in res) / seconds / n:.2f} "
          f"per NPC) | failed steps {sum(r['errors'] for r in res)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Headless crowd stress test")
    ap.add_argument("npcs", type=int, nargs="+")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--nvidia-settings", action="store_true", help="NVIDIA's default settings instead of the calm profile")
    args = ap.parse_args()
    for n in args.npcs:
        stress(n, args.procs, args.hz, args.seconds, calm=not args.nvidia_settings)


if __name__ == "__main__":
    main()
