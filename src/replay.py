"""Dump a self-play game as a watchable `.hbr2` so you can SEE how training is going.

`train.py --replay-every N` calls this every N iters. We play one game in the Rust core and
record each player's per-decision input mask; then (if Node + HaxballGym's `record_replay.js`
are available) we replay those inputs in an offline node-haxball sandbox to write a real `.hbr2`
loadable in any Haxball replay viewer. Without Node we still write the `.trace.json` and tell you
how to convert it. The `.hbr2` step needs Node because node-haxball is the actual game recorder.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from haxballgym import DefaultObs, KickoffMutator, TransitionEngine, stadium_text


def bins_to_mask(dx: int, dy: int, kick: int) -> int:
    """World-frame action bins {-1,0,1} -> Haxball 5-bit input mask (UP1 DOWN2 LEFT4 RIGHT8 KICK16)."""
    m = 0
    m |= 8 if dx > 0 else 4 if dx < 0 else 0  # RIGHT / LEFT
    m |= 1 if dy > 0 else 2 if dy < 0 else 0  # UP / DOWN
    m |= 16 if kick else 0
    return m


@torch.no_grad()
def dump_trace(model, *, stadium, n_red, n_blue, tick_skip, n_decisions=300, device="cpu"):
    """Play one self-play game (model controls every player) and return the input-trace dict."""
    P = n_red + n_blue
    stad = stadium or "classic"
    eng = TransitionEngine(1, n_red, n_blue, stadium=stadium, tick_skip=tick_skip)
    obs_builder = DefaultObs(pos_coef="auto") if stadium else DefaultObs()
    state = KickoffMutator().reset_all(eng)
    obs_builder.reset(state)
    spawns = np.asarray(state.player_pos)[0].tolist()  # record_replay.js maps haxball players by spawn
    decisions = []
    for _ in range(n_decisions):
        o = np.asarray(obs_builder.build_obs(state))
        flat = torch.as_tensor(o.reshape(-1, o.shape[-1]), dtype=torch.float32, device=device)
        bins = model.act(flat).cpu().numpy().reshape(P, 3)
        masks, eng_acts = [], np.zeros((1, P, 3), np.int64)
        for p in range(P):
            dx, dy, kick = int(bins[p, 0]) - 1, int(bins[p, 1]) - 1, int(bins[p, 2])
            wdx = dx if p < n_red else -dx  # blue un-mirrors dx (matches Env._mirror_x / training)
            eng_acts[0, p] = (wdx, dy, kick)
            masks.append(bins_to_mask(wdx, dy, kick))
        decisions.append(masks)
        state = eng.step(eng_acts)
    return {
        "hbs": stadium_text(stad),
        "n_red": n_red,
        "n_blue": n_blue,
        "tick_skip": tick_skip,
        "spawns": spawns,
        "decisions": decisions,
    }


def dump_replay(
    model,
    out_hbr2,
    *,
    stadium,
    n_red,
    n_blue,
    tick_skip,
    record_js,
    n_decisions=300,
    device="cpu",
    background=False,
):
    """Write `out_hbr2` (or just the trace if no Node/record_replay.js). Best-effort: never raises.
    `background=True` launches the Node recorder fire-and-forget (don't block the training loop)."""
    out_hbr2 = Path(out_hbr2)
    trace_path = out_hbr2.with_suffix(".trace.json")
    trace_path.write_text(
        json.dumps(
            dump_trace(
                model,
                stadium=stadium,
                n_red=n_red,
                n_blue=n_blue,
                tick_skip=tick_skip,
                n_decisions=n_decisions,
                device=device,
            )
        )
    )
    if not record_js or not Path(record_js).exists():
        return False, f"trace at {trace_path} (set --record-js <path/to/record_replay.js> for .hbr2)"
    cmd = ["node", str(record_js), str(trace_path), str(out_hbr2)]
    try:
        if background:  # fire-and-forget: node reads the trace async; don't stall training
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "(recording in background)"
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout)[-300:]
        trace_path.unlink()
        return True, str(out_hbr2)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
