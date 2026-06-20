"""Headless evaluation: WazBot vs ChaseBot (or vs itself) over many parallel games.

    uv run src/eval.py                 # WazBot vs ChaseBot
    uv run src/eval.py --self --n 512 --steps 800

Goals come from `state.scored` (who conceded each step) — reward-independent, so the tally
is right no matter how the reward is configured.
"""

from __future__ import annotations

import argparse

import numpy as np

from bot import ChaseBot, NeuralBot
from haxballgym import BLUE, RED, make_default_env


def play(red, blue_is_chase: bool, n_envs: int, steps: int):
    env = make_default_env(n_envs, 1, 1)
    obs = env.reset()
    blue = ChaseBot(seed=0) if blue_is_chase else NeuralBot(greedy=True)
    rg = bg = 0
    for _ in range(steps):
        a_red = red.act(obs[:, 0:1]).reshape(n_envs, 1, 3)
        a_blue = (
            blue.act(env.prev_state, [1]) if blue_is_chase else blue.act(obs[:, 1:2]).reshape(n_envs, 1, 3)
        )
        obs, rew, term, trunc = env.step(np.concatenate([a_red, a_blue], axis=1))
        sc = env.prev_state.scored
        rg += int((sc == BLUE).sum())  # blue conceded -> red (WazBot) scored
        bg += int((sc == RED).sum())  # red conceded -> blue scored
    return rg, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="parallel games")
    ap.add_argument("--steps", type=int, default=600, help="decisions per game")
    ap.add_argument("--self", action="store_true", help="WazBot vs WazBot instead of vs ChaseBot")
    args = ap.parse_args()

    rg, bg = play(
        NeuralBot(greedy=True),
        blue_is_chase=not args.self,
        n_envs=args.n,
        steps=args.steps,
    )
    name = "WazBot" if args.self else "ChaseBot"
    print(f"WazBot {rg} - {bg} {name}   (net {rg - bg:+d} over {args.n} games x {args.steps} steps)")


if __name__ == "__main__":
    main()
