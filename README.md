# WazBot

A solid starting point for building a [Haxball](https://www.haxball.com/) bot with
[**HaxballGym**](https://github.com/HaxballGym/HaxballGym) (the
[`haxballgym`](https://pypi.org/project/haxballgym/) package).

It ships a trained bot (`models/model.pt`), a self-contained self-play **trainer**, a
scripted **ChaseBot** opponent, a headless **evaluator**, and a guide — including which
training techniques are actually worth your time.

HaxballGym is a batched, RLGym-v2-style env over a headless Rust engine that's **bit-exact
to the real Haxball game**, so what you train transfers straight to a real room.

---

## Setup

Uses [**uv**](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Train

Self-play PPO — the learner plays frozen copies of itself (a snapshot pool), which keeps
training stable. It saves periodic checkpoints to `models/ckpts/`.

```bash
uv run src/train.py                                  # 1v1 on classic
uv run src/train.py --iters 10000 --stadium big --red 3 --blue 3 --device mps
```

The live "net vs chase" print is only a rough sanity signal — it **saturates and is noisy**,
so don't trust it to pick the best checkpoint. Instead, after training:

```bash
uv run src/select.py   # round-robins the saved checkpoints head-to-head, copies the strongest -> model.pt
```

That's the model `eval.py`, `NeuralBot`, and the room loader use.

## Evaluate

```bash
uv run src/eval.py            # WazBot vs ChaseBot, prints the goal tally
uv run src/eval.py --self     # WazBot vs WazBot
```

## Watch / play against it in a real room

The bot deploys into a **real Haxball room** (the engine is bit-exact, so the policy just
works). Use HaxballGym's room loader — one command, pointed at this repo's model:

```bash
# from the HaxballGym repo (needs Node + a token from haxball.com/headlesstoken)
HEADLESS_TOKEN=<token> uv run headless-bot/play.py /path/to/wazbot-public/models/model.pt
```

It prints a room link; join from the official client and play the bot 1v1.

---

## What actually helps (and what doesn't)

We ran a lot of ablations. The blunt summary: **the biggest lever is training the simple
recipe for longer** — most of the fancy additions barely moved the needle.

**Worth it** — and what `src/train.py` does:

- **PPO self-play** — every player is the live policy, learning from its own games.
- **A goal-dominated reward**: dense move-to-ball + ball-to-goal shaping, plus a sparse
  goal reward weighted far higher (HaxballGym's `make_default_env` default).
- **LR / entropy annealing**, and just **training longer**.
- A **snapshot pool** (mixing in frozen past copies as opponents) is a cheap refinement
  if you want to push further — it curbs strategy cycling.

**Skip it** (didn't help, or hurt, in our runs):

- **Replay / human imitation (BC, GAIL)** — no clear gain over self-play trained longer.
- **y-symmetry augmentation** — actively hurt. **PFSP** over a plain pool, **distillation**,
  **bigger networks**, **action-stacking**, alternative obs builders — marginal to negative.

### Adding replays (optional — not used by the shipped model)

To learn from human `.hbr2` replays: parse them into `(observation, action)` pairs, then
either behavior-clone a prior and warm-start PPO from it, or run GAIL (a discriminator that
rewards human-like play). We **excluded replays from the shipped model** — they didn't beat
plain self-play, and a clean self-play model has no human-data dependency.

---

## Layout

| File | What it is |
|---|---|
| `src/train.py`  | Self-play PPO trainer (snapshot pool); saves periodic checkpoints. |
| `src/select.py` | Round-robins the checkpoints, copies the strongest to `model.pt`. |
| `src/model.py`  | The policy network (heads sized from the env; deploy-compatible). |
| `src/bot.py`    | `ChaseBot` (scripted, reads the state) and `NeuralBot` (loads `models/`). |
| `src/eval.py`   | Headless goal tally. |
| `models/`       | The shipped trained policy (`model.pt` + `meta.json`). |

Happy training!
