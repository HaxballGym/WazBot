import logging
from dataclasses import dataclass

import numpy as np
from haxballgym import make
from haxballgym.gym import Gym
from haxballgym.utils.action_parsers import DefaultAction
from haxballgym.utils.obs_builders import DefaultObs
from haxballgym.utils.reward_functions.common_rewards import misc_rewards
from haxballgym.utils.terminal_conditions import common_conditions
from haxballgym_tools.sb3_utils import SB3SingleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize
from ursina import Keys, held_keys
from ursinaxball import Game
from ursinaxball import common_values as cv

frame_skip = 10
half_life_seconds = 5


@dataclass
class InputPlayer:
    left: list[str]
    right: list[str]
    up: list[str]
    down: list[str]
    shoot: list[str]


input_player_1 = InputPlayer(
    left=[Keys.left_arrow],
    right=[Keys.right_arrow],
    up=[Keys.up_arrow],
    down=[Keys.down_arrow],
    shoot=["x"],
)


def action_handle(actions_player_output: list[int], inputs_player: InputPlayer):
    actions_player_output = [1, 1, 0]
    for key, value in held_keys.items():
        if value != 0:
            if key in inputs_player.left:
                actions_player_output[0] -= 1
            if key in inputs_player.right:
                actions_player_output[0] += 1
            if key in inputs_player.up:
                actions_player_output[1] += 1
            if key in inputs_player.down:
                actions_player_output[1] -= 1
            if key in inputs_player.shoot:
                actions_player_output[2] += 1
    return actions_player_output


# Customize this function in case you want your games against your bot saved
def recording_condition(gym_env: Gym) -> bool:
    game = gym_env._match._game
    if game.score.red + game.score.blue > 0 and game.score.time > 10:
        return False

    return False


def create_gym_env_play():
    game = Game(
        stadium_file=cv.BaseMap.CLASSIC,
        folder_rec="./recordings/",
        logging_level=logging.DEBUG,
        enable_renderer=True,
        enable_vsync=True,
        enable_recorder=False,
    )
    gym_env = make(
        game=game,
        reward_fn=misc_rewards.ConstantReward(),
        terminal_conditions=common_conditions.GoalScoredCondition(),
        obs_builder=DefaultObs(),
        action_parser=DefaultAction(),
        team_size=1,
        tick_skip=0,
    )
    return gym_env


gym_env = create_gym_env_play()

fps = 60 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))

# wrap the environment with the single instance wrapper
env = SB3SingleInstanceEnv(env=gym_env, recording_condition=recording_condition)
env = VecCheckNan(env)
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=False, gamma=gamma)

model = PPO.load(
    "logs/model.zip",
    env,
    custom_objects=dict(n_envs=env.num_envs, _last_obs=None),
    device="auto",
)


vec_env = model.get_env()
obs = vec_env.reset()

while True:
    done = [False, False]
    steps = 0
    actions = [[1, 1, 0], [1, 1, 0]]
    actions[0] = action_handle(actions[0], input_player_1)
    if steps % (frame_skip + 1) == 0:
        action_bot, _ = model.predict(obs, deterministic=True)
        actions[1] = action_bot[1]
    obs, reward, done, info = vec_env.step(actions)
    steps += 1
    if done.any():
        obs = vec_env.reset()
