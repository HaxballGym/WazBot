"""Train a WazBot by self-play PPO — self-contained, one command:

    uv run src/train.py                                  # 1v1 on classic, ~45 min on a laptop CPU
    uv run src/train.py --iters 6000 --stadium big --red 3 --blue 3

The learner controls RED; the opponent (BLUE) is a FROZEN copy of the policy — usually the
current one, sometimes an older snapshot from a pool. Training against frozen past selves
(instead of a live mirror of itself) is what keeps self-play stable: a plain live mirror
co-adapts and collapses. The observation is goal-relative, so a red-trained net plays either
side at deploy time. Reward = HaxballGym's default (move-to-ball + ball-to-goal shaping with a
goal reward weighted far higher). Saves the best checkpoint to `models/model.pt` (+ `meta.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bot import ChaseBot
from haxballgym import BLUE, RED, make_default_env
from model import Policy
from replay import dump_replay

# PPO / GAE hyperparameters (CleanRL-style; the defaults that worked best in our ablations).
T, EPOCHS, N_MB = 64, 4, 4  # rollout length, update epochs, minibatches per epoch
CLIP, VF_COEF, MAX_GRAD, LAM = 0.2, 0.5, 0.5, 0.95
LR, ENT0, ENT1 = 3e-4, 0.01, 0.01  # constant entropy — sustained exploration prevents collapse
HALF_LIFE_S = 5.0  # discount: value 5 s of play at half weight
# Snapshot pool: add every 20 iters, keep 40 (spans ~800 iters), face a PAST self 50% of games.
# High diversity is the anti-collapse lever — the learner can't co-adapt with its current mirror.
SNAPSHOT_EVERY, POOL_SIZE, POOL_PROB = 20, 40, 0.5


@torch.no_grad()
def eval_vs_chase(model, dev, n_red, n_blue, stadium, tick_skip, n_envs=256, steps=400):
    """Net goals of `model` (red) vs a scripted ChaseBot (blue). Goals via `state.scored`
    (reward-independent). A decent bot is strongly positive; a random one is negative."""
    env = make_default_env(n_envs, n_red, n_blue, stadium=stadium, tick_skip=tick_skip)
    obs = env.reset()
    chase = ChaseBot(seed=0)
    P, od = n_red + n_blue, obs.shape[-1]
    red_i, blue_i = np.arange(n_red), np.arange(n_red, P)
    rg = bg = 0
    for _ in range(steps):
        red = model.act(torch.as_tensor(obs[:, red_i].reshape(-1, od), dtype=torch.float32, device=dev))
        full = np.empty((n_envs, P, 3), dtype=np.int64)
        full[:, red_i] = red.cpu().numpy().reshape(n_envs, n_red, 3)
        full[:, blue_i] = chase.act(env.prev_state, blue_i)
        obs, rew, term, trunc = env.step(full)
        sc = env.prev_state.scored
        rg += int((sc == BLUE).sum())  # blue conceded -> red scored
        bg += int((sc == RED).sum())
    return rg - bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--envs", type=int, default=512, help="parallel games")
    ap.add_argument("--red", type=int, default=1)
    ap.add_argument("--blue", type=int, default=1)
    ap.add_argument("--stadium", default=None, help="bundled name or .hbs path (default classic)")
    ap.add_argument("--tick-skip", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible runs on a given machine)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--save-every", type=int, default=1000, help="also snapshot ckpts/it{N}.pt every N iters")
    ap.add_argument(
        "--replay-every", type=int, default=0, help="dump a watchable replay every N iters (0=off)"
    )
    ap.add_argument(
        "--record-js", default=None, help="path to HaxballGym's record_replay.js (for .hbr2 output)"
    )
    args = ap.parse_args()

    dev = torch.device(args.device)
    # Seed everything so a run is reproducible on a given machine: torch (model init + action
    # sampling), global numpy (the PPO minibatch shuffle), and our pool-selection rng. (RL across
    # different machines is never bit-identical — thread count / float order — but this kills the
    # big run-to-run variance.)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_default_env(args.envs, args.red, args.blue, tick_skip=args.tick_skip, stadium=args.stadium)
    P = args.red + args.blue
    obs_dim = env.obs_dim
    n_acts = env.action_space  # (3, 3, kick_n) — sizes come from the env, nothing hardcoded
    gamma = math.exp(math.log(0.5) / ((60 / args.tick_skip) * HALF_LIFE_S))

    # Slot bookkeeping: players are flattened to (env*P + local). Learner = red columns, frozen
    # opponent = blue columns. Only learner slots are trained on.
    slots = np.arange(args.envs * P).reshape(args.envs, P)
    learner_idx = slots[:, : args.red].reshape(-1)
    opp_idx = slots[:, args.red :].reshape(-1)
    learner_t = torch.as_tensor(learner_idx, device=dev)
    opp_t = torch.as_tensor(opp_idx, device=dev)
    B = learner_idx.size

    model = Policy(obs_dim, n_acts).to(dev)
    opp = Policy(obs_dim, n_acts).to(dev)
    opp.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pool: list[dict] = []  # frozen past snapshots (cpu state_dicts)

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"self-play PPO (snapshot pool) | {args.red}v{args.blue} {args.stadium or 'classic'} | "
        f"obs_dim={obs_dim} n_acts={n_acts} train_slots={B} gamma={gamma:.4f} device={dev} iters={args.iters}"
    )

    S_obs = torch.zeros(T, B, obs_dim, device=dev)
    S_act = torch.zeros(T, B, 3, dtype=torch.long, device=dev)
    S_logp = torch.zeros(T, B, device=dev)
    S_val = torch.zeros(T, B, device=dev)
    S_rew = torch.zeros(T, B, device=dev)
    S_done = torch.zeros(T, B, device=dev)

    obs = env.reset().reshape(args.envs * P, obs_dim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -(10**9)
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
        frac = (it - 1) / max(1, args.iters - 1)
        for g in opt.param_groups:
            g["lr"] = LR * (1 - frac) + (LR / 3) * frac  # anneal LR -> LR/3
        ent_coef = ENT0 * (1 - frac) + ENT1 * frac

        # pick the frozen opponent for this iter: current copy (stay sharp) or a past snapshot
        opp.load_state_dict(
            pool[rng.integers(len(pool))] if (pool and rng.random() < POOL_PROB) else snapshot()
        )

        # --- rollout (learner=red trains, opponent=blue is frozen) ---
        for t in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            with torch.no_grad():
                act, logp, val = model.act_train(o[learner_t])
                o_act = opp.act(o[opp_t])
            S_obs[t], S_act[t], S_logp[t], S_val[t] = o[learner_t], act, logp, val
            full = np.empty((args.envs * P, 3), dtype=np.int64)
            full[learner_idx] = act.cpu().numpy()
            full[opp_idx] = o_act.cpu().numpy()
            nobs, rew, term, trunc = env.step(full.reshape(args.envs, P, 3))
            obs = nobs.reshape(args.envs * P, obs_dim)
            done = np.repeat((term | trunc).astype(np.float32), P)
            S_rew[t] = torch.as_tensor(rew.reshape(-1)[learner_idx], dtype=torch.float32, device=dev)
            S_done[t] = torch.as_tensor(done[learner_idx], device=dev)

        # --- GAE(lambda) advantages ---
        with torch.no_grad():
            last_val = model._logits(torch.as_tensor(obs, dtype=torch.float32, device=dev)[learner_t])[1]
        adv = torch.zeros(T, B, device=dev)
        gae = torch.zeros(B, device=dev)
        for t in reversed(range(T)):
            nonterm = 1.0 - S_done[t]
            nextval = last_val if t == T - 1 else S_val[t + 1]
            delta = S_rew[t] + gamma * nextval * nonterm - S_val[t]
            gae = delta + gamma * LAM * nonterm * gae
            adv[t] = gae
        ret = adv + S_val

        # --- PPO update ---
        bo, ba, bl = S_obs.reshape(-1, obs_dim), S_act.reshape(-1, 3), S_logp.reshape(-1)
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx = np.arange(T * B)
        mb = len(idx) // N_MB
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                mi = torch.as_tensor(idx[s : s + mb], device=dev)
                logp, ent, val = model.evaluate(bo[mi], ba[mi])
                ratio = (logp - bl[mi]).exp()
                a = badv[mi]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                vl = 0.5 * (val - bret[mi]).pow(2).mean()
                loss = pl + VF_COEF * vl - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                opt.step()

        if it % SNAPSHOT_EVERY == 0:  # grow the opponent pool
            pool.append(snapshot())
            if len(pool) > POOL_SIZE:
                pool.pop(0)

        if args.save_every and it % args.save_every == 0:  # periodic checkpoints for later selection
            (out / "ckpts").mkdir(exist_ok=True)
            torch.save(model.state_dict(), out / "ckpts" / f"it{it}.pt")

        if args.replay_every and it % args.replay_every == 0:  # watchable .hbr2 of the current bot
            (out / "replays").mkdir(exist_ok=True)
            dump_replay(
                model,
                out / "replays" / f"it{it}.hbr2",
                stadium=args.stadium,
                n_red=args.red,
                n_blue=args.blue,
                tick_skip=args.tick_skip,
                record_js=args.record_js,
                background=True,
            )

        # --- eval + checkpoint ---
        if it % 50 == 0 or it == args.iters:
            net = eval_vs_chase(model, dev, args.red, args.blue, args.stadium, args.tick_skip)
            sps = (it * T * B) / (time.perf_counter() - t0)

            def meta(n):  # arch + which map/teams, plus this checkpoint's score
                return json.dumps(
                    {
                        "obs_dim": obs_dim,
                        "n_acts": list(n_acts),
                        "hidden": 256,
                        "depth": 2,
                        "n_red": args.red,
                        "n_blue": args.blue,
                        "stadium": args.stadium or "classic",
                        "net_vs_chase": n,
                    }
                )

            # `model.pt` = best-by-chase; `model_final.pt` = latest weights. net-vs-chase
            # saturates/is noisy, so keep BOTH and let an Elo run vs real bots pick the winner.
            torch.save(model.state_dict(), out / "model_final.pt")
            (out / "meta_final.json").write_text(meta(net))
            tag = ""
            if net > best:
                best = net
                torch.save(model.state_dict(), out / "model.pt")
                (out / "meta.json").write_text(meta(net))
                tag = "  <- saved best"
            print(f"it {it:4d} | {sps / 1e3:4.0f}k steps/s | net vs chase {net:+d} (best {best:+d}){tag}")

    print(f"done. best-by-chase {best:+d} -> {out / 'model.pt'} ; final -> {out / 'model_final.pt'}")


if __name__ == "__main__":
    main()
