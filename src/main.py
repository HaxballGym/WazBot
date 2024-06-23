import logging
from pathlib import Path

import numpy as np
from haxballgym import make
from haxballgym.utils.action_parsers import DefaultAction
from haxballgym.utils.obs_builders import DefaultObs
from haxballgym.utils.reward_functions import (
    CombinedReward,
    velocity_reward,
)
from haxballgym.utils.terminal_conditions import common_conditions
from haxballgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize
from ursinaxball import Game
from ursinaxball import common_values as cv

FRAME_SKIP = 10
HALF_LIFE_SECONDS = 5
NB_GAMES = 8


def create_gym_env(render: bool):
    game = Game(
        stadium_file=cv.BaseMap.CLASSIC,
        logging_level=logging.DEBUG,
        enable_renderer=render,
        enable_vsync=False,
        enable_recorder=False,
    )
    gym_env = make(
        game=game,
        reward_fn=CombinedReward(
            (
                velocity_reward.VelocityPlayerToBallReward(),
                velocity_reward.VelocityBallToGoalReward(
                    stadium=game.stadium_store, own_goal=False
                ),
            )
        ),
        terminal_conditions=(
            common_conditions.TimeoutCondition(int(1 * 60 * 60 / FRAME_SKIP)),
            common_conditions.GoalScoredCondition(),
        ),
        obs_builder=DefaultObs(),
        action_parser=DefaultAction(),
        team_size=1,
        tick_skip=FRAME_SKIP,
    )
    return gym_env._match


def create_env(render: bool):
    # Set the first array argument as True to visualize progress.
    # Warning: performance are greatly impacted. I recommend to train without a GUI
    NB_GAMES = 8
    gym_env = [create_gym_env(False)] + [
        create_gym_env(False) for _ in range(NB_GAMES - 1)
    ]

    fps = 60 / FRAME_SKIP
    gamma = np.exp(np.log(0.5) / (fps * HALF_LIFE_SECONDS))

    env = SB3MultipleInstanceEnv(gym_env)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    return env


def model_new():
    env = create_env(False)

    fps = 60 / FRAME_SKIP
    gamma = np.exp(np.log(0.5) / (fps * HALF_LIFE_SECONDS))

    # create a PPO instance and start learning
    model = PPO(
        "MlpPolicy",
        env,
        n_epochs=32,  # PPO calls for multiple epochs
        learning_rate=1e-5,  # Around this is fairly common for PPO
        ent_coef=0.01,  # From PPO Atari
        vf_coef=1.0,  # From PPO Atari
        gamma=gamma,  # Gamma as calculated using half-life
        verbose=3,  # Print out all the info as we're going
        batch_size=4096,  # Batch size as high as possible within reason
        n_steps=4096,  # Number of steps to perform before optimizing network
        tensorboard_log="policy/",  # `tensorboard --logdir policy` in terminal
        device="auto",  # Uses GPU if available
    )

    return model


def model_continue(path_model: Path):
    env = create_env(False)

    # load a model file
    model = PPO.load(
        path_model,
        env,
        custom_objects=dict(
            n_envs=env.num_envs, _last_obs=None
        ),  # Need this to change number of agents
        device="auto",  # Need to set device again (if using a specific one)
    )
    env.reset()  # Important when loading models, SB3 does not do this for you

    return model, checkpoint_callback


def main():
    model = model_new()
    # model = model_continue(Path("logs/model.zip"))

    # Save a checkpoint every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=int(100_000 / NB_GAMES),
        save_path="logs/",
        name_prefix="multi",
    )

    model.learn(100_000_000, checkpoint_callback)


if __name__ == "__main__":
    main()
