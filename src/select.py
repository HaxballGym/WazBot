"""Pick the strongest checkpoint from a training run, then make it the shipped model.

    uv run src/select.py            # round-robin models/ckpts/*.pt (+ model_final.pt), winner -> model.pt

net-vs-chase saturates and is noisy, so `train.py` can't reliably pick the best checkpoint
while training. This plays every saved checkpoint against every other (head-to-head, both
colors; goals counted from `state.scored`, so it's reward-independent) and copies the one with
the best overall goal share to `models/model.pt` (+ `meta.json`). No external bots involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from haxballgym import BLUE, RED, make_default_env
from model import Policy

MODELS = Path(__file__).resolve().parent.parent / "models"


def load(path: Path):
    """Rebuild a Policy from a checkpoint, inferring the architecture from the weights."""
    sd = torch.load(path, map_location="cpu")
    obs_dim = sd["trunk.0.weight"].shape[1]
    hidden = sd["trunk.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".weight"))
    nk = sd["head_k.weight"].shape[0]
    m = Policy(obs_dim, (3, 3, nk), hidden, depth)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def h2h(a, b, n_envs=256, steps=400):
    """Goals (a, b) with `a`=red, `b`=blue over one set of games."""
    env = make_default_env(n_envs, 1, 1)
    obs = env.reset()
    od = obs.shape[-1]
    ag = bg = 0
    for _ in range(steps):
        ra = a.act(torch.as_tensor(obs[:, 0:1].reshape(-1, od), dtype=torch.float32)).numpy()
        rb = b.act(torch.as_tensor(obs[:, 1:2].reshape(-1, od), dtype=torch.float32)).numpy()
        obs, rew, term, trunc = env.step(np.stack([ra, rb], axis=1).reshape(n_envs, 2, 3))
        sc = env.prev_state.scored
        ag += int((sc == BLUE).sum())  # blue conceded -> a (red) scored
        bg += int((sc == RED).sum())
    return ag, bg


def main():
    paths = sorted((MODELS / "ckpts").glob("it*.pt"), key=lambda p: int(p.stem[2:]))
    for extra in ("model.pt", "model_final.pt"):  # also consider best-by-chase + final
        if (MODELS / extra).exists():
            paths.append(MODELS / extra)
    if len(paths) < 2:
        print("need >=2 checkpoints in models/ckpts/ (train with --save-every)")
        return
    nets = {p.stem: load(p) for p in paths}
    gf = dict.fromkeys(nets, 0)
    ga = dict.fromkeys(nets, 0)
    names = list(nets)
    for i, x in enumerate(names):  # round-robin, each pair once (both colors)
        for y in names[i + 1 :]:
            xr, yb = h2h(nets[x], nets[y])  # x red, y blue
            yr, xb = h2h(nets[y], nets[x])  # swap colors
            gf[x] += xr + xb
            ga[x] += yb + yr
            gf[y] += yr + yb
            ga[y] += xb + xr

    ranked = sorted(names, key=lambda k: gf[k] / max(gf[k] + ga[k], 1), reverse=True)
    print("checkpoint            goal-share")
    for k in ranked:
        print(f"  {k:18s}  {gf[k] / max(gf[k] + ga[k], 1):.3f}")
    win = ranked[0]
    torch.save(nets[win].state_dict(), MODELS / "model.pt")
    meta = json.loads((MODELS / "meta_final.json").read_text())  # arch is identical across ckpts
    meta["selected"] = win
    (MODELS / "meta.json").write_text(json.dumps(meta))
    print(f"\nstrongest = {win} -> copied to models/model.pt")


if __name__ == "__main__":
    main()
