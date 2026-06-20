"""Two bots over a HaxballGym env.

- `ChaseBot`  — scripted baseline: run at the ball, kick when close. It reads the raw
  game **state** (`ball_pos` / `player_pos`), so there are no magic observation indices.
- `NeuralBot` — loads the trained policy (`models/model.pt` + `meta.json`).

`ChaseBot.act(state, players)` takes a `GameState` and the player columns to control.
`NeuralBot.act(obs)` takes an egocentric observation block (what the policy is trained on).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ChaseBot:
    """Heuristic: steer toward the ball (with a little angular jiggle so it isn't on rails),
    kick when within `kick_dist`. Reads positions from the state — map/obs agnostic."""

    def __init__(self, kick_dist: float = 34.0, jiggle: float = 0.25, seed: int | None = None):
        self.kick_dist = kick_dist
        self.jiggle = jiggle
        self.rng = np.random.default_rng(seed)

    def act(self, state, players) -> np.ndarray:
        """state: GameState; players: index array of columns to control -> bins (N, K, 3)."""
        ball = state.ball_pos[:, None, :]  # (N, 1, 2)
        pos = state.player_pos[:, players, :]  # (N, K, 2)
        d = ball - pos
        dist = np.hypot(d[..., 0], d[..., 1])
        ang = np.arctan2(d[..., 1], d[..., 0]) + self.rng.normal(0.0, self.jiggle, dist.shape)
        dx, dy = np.cos(ang), np.sin(ang)
        bx = np.where(dx > 0.33, 2, np.where(dx < -0.33, 0, 1))
        by = np.where(dy > 0.33, 2, np.where(dy < -0.33, 0, 1))
        kick = (dist < self.kick_dist).astype(np.int64)
        return np.stack([bx, by, kick], -1).astype(np.int64)


class NeuralBot:
    """The trained WazBot. Greedy (deterministic) by default; `greedy=False` samples."""

    def __init__(self, model_dir: str | Path | None = None, greedy: bool = True):
        import torch  # local import so ChaseBot stays torch-free

        from model import Policy

        model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "models"
        meta = json.loads((model_dir / "meta.json").read_text())
        self.greedy = greedy
        self._torch = torch
        self.model = Policy(meta["obs_dim"], meta["n_acts"], meta["hidden"], meta["depth"])
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs (..., obs_dim) -> bins (M, len(n_acts))."""
        o = self._torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=self._torch.float32)
        return self.model.act(o, greedy=self.greedy).numpy().astype(np.int64)
