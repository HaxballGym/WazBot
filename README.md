# WazBot

This repository is made to provide you with a solid start in HaxballGym.

## Installation

Before installing dependencies with pip, you must first use the appropriate version of pytorch for your system.

The list of wheels is available [here](https://download.pytorch.org/whl/torch/)

### `pip`

`pip install -r requirement.txt`

### Poetry

There is a bug with the version 0.21.0 of gym that makes the installation on Poetry unsmooth. With additional commands, it is possible to still complete the installation.

```bash
poetry install # install everything, except gym who should make an error
poetry run pip install gym==0.21.0
poetry install
```

## Files

### main.py

This is the main file for training. With the reward function, you are able to train a pretty capable bot.

### reward_testing.py

This file allows you to test your reward functions: you are able to play and see in the terminal whether the reward function you defined works for you.

### play_model.py

After your training, you are able to run this file to play against your bot! With this, you can find potential weaknesses from your agent to learn from your mistakes and continue improving.

### play_model_bot.py

After your training, you are able to run this file to see your agent train against a regular bot.

Happy training!
