#!/usr/bin/env python
import sys
import gym
import numpy as np
from util import (Trainer, ObservationTransform,
                  HorizonObservationWrapper, PhaseRewardWrapper,
                  RandomActionWrapper, RelativeActionWrapper, OurActionWrapper)
from stable_baselines3 import PPO
from gym import spaces, ActionWrapper


# env = gym.make("reference_environment:reference-environment-v0")

# Train an RL agent on the environment
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)


# ## Training on horizon observations
# env = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=30,
#                               transform_name="Standard")
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)

### Testing DDPG
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
env_action = RelativeActionWrapper(gym.make("reference_environment:reference-environment-v0"))
env_horizon = HorizonObservationWrapper(env_action,
                              horizon_length=int(sys.argv[2]),
                              transform_name=str(sys.argv[1]))
env = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)


# ### DDPG Noise
# ### Try increasing the noise when retraining.
# ### Try less noise based on the policy plot.
# n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
# # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#
# model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./logs",
#             gamma=0.99,
#             learning_rate=0.0003,
#             )
# # model = DDPG.load("Model_DDPG_FS_30.zip")
# # model.learning_rate = 0.0003
# # model.gamma = 0.99
# # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.05*np.ones(n_actions))
# # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.075 * np.ones(n_actions))
# # model.action_noise = action_noise
# trainer = Trainer(env)
# trainer.retrain_rl(model, episodes=1000000, ident=str(sys.argv[1]+"_"+sys.argv[2]))
