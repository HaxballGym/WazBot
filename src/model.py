"""The policy network — a small actor-critic MLP.

Three categorical heads (dx, dy, kick) over a shared trunk, plus a value head. The head
*sizes* come from the env's action space (`env.action_space` -> `(3, 3, kick_n)`), so
nothing about the action layout is hardcoded — but the weight names (`trunk.0/2`,
`head_x/y/k`) and depth-2 trunk match HaxballGym's `headless-bot` exporter, so a trained
`model.pt` deploys to a real room with no extra work. `meta.json` records the shapes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_acts: Sequence[int] = (3, 3, 2),
        hidden: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        self.n_acts = tuple(n_acts)
        nx, ny, nk = self.n_acts
        layers, d = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_x = nn.Linear(hidden, nx)  # dx bin
        self.head_y = nn.Linear(hidden, ny)  # dy bin
        self.head_k = nn.Linear(hidden, nk)  # kick: 0/1 (release/hold) or 0/1/2 (+rocket)
        self.value = nn.Linear(hidden, 1)

    def _logits(self, obs):
        h = self.trunk(obs)
        return (self.head_x(h), self.head_y(h), self.head_k(h)), self.value(h).squeeze(-1)

    def act_train(self, obs):
        """Sample actions + return (actions, summed log-prob, value) for a rollout step."""
        logits, v = self._logits(obs)
        dists = [Categorical(logits=lg) for lg in logits]
        a = torch.stack([d.sample() for d in dists], -1)
        logp = sum(d.log_prob(a[:, i]) for i, d in enumerate(dists))
        return a, logp, v

    def evaluate(self, obs, acts):
        """Log-prob + entropy of given actions under the current policy (PPO update)."""
        logits, v = self._logits(obs)
        dists = [Categorical(logits=lg) for lg in logits]
        logp = sum(d.log_prob(acts[:, i]) for i, d in enumerate(dists))
        ent = sum(d.entropy() for d in dists)
        return logp, ent, v

    @torch.no_grad()
    def act(self, obs, greedy: bool = False):
        """obs (M, obs_dim) float tensor -> action bins (M, 3) long tensor."""
        logits, _ = self._logits(obs)
        if greedy:
            return torch.stack([lg.argmax(-1) for lg in logits], -1)
        return torch.stack([Categorical(logits=lg).sample() for lg in logits], -1)
