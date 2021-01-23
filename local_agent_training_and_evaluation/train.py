#!/usr/bin/env python
import gym
import numpy as np
from util import Trainer, ObservationTransform, HorizonObservationWrapper, PhaseRewardWrapper, OurActionWrapper
from stable_baselines3 import PPO


# env = gym.make("reference_environment:reference-environment-v0")

# Train an RL agent on the environment
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)


### Training on horizon observations
# env = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=50,
#                               transform_name="Standard")
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1,episodes_per_model=20000)


### Testing phase reward wrapper
# env=PhaseRewardWrapper(gym.make("reference_environment:reference-environment-v0"), phase="Peak")
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=1000)


### Test nested wrappers
# env_horizon = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=20,
#                               transform_name="Standard")
# env = PhaseRewardWrapper(env_horizon, phase="Peak")
# trainer = Trainer(env)
# # trainer.train_rl(models_to_train=1,episodes_per_model=1000)                   # Begin Training
# trainer.retrain_rl(model=PPO.load("logs/best_model_peak_20"), episodes=1000)   # Retraining


### Test training on peak then full
# env_action = OurActionWrapper(gym.make("reference_environment:reference-environment-v0"))
# env_horizon = HorizonObservationWrapper(env_action,
#                               horizon_length=30,
#                               transform_name="Standard")
# env_peak = PhaseRewardWrapper(env_horizon, phase="Peak")          # Set Phase to Peak
# env_full = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
#
#
# trainer = Trainer(env_peak)
# # trainer.train_rl(models_to_train=1,episodes_per_model=3000)       # Begin Training
# model = PPO.load("logs/best_model_peak_30")                               # Load best model
# model.learning_rate = 0.0003
# #
# trainer = Trainer(env_full)
# trainer.retrain_rl(model=model, episodes=50000)                    # Re-train on full phase


### Testing DDPG
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
env_action = OurActionWrapper(gym.make("reference_environment:reference-environment-v0"))
env_horizon = HorizonObservationWrapper(env_action,
                              horizon_length=30,
                              transform_name="Standard")
env = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)


### DDPG Noise
### Try increasing the noise when retraining.
### Try less noise based on the policy plot.
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./logs/",
            gamma=0.99,
            learning_rate=0.0003,
            )
# model = DDPG.load("Model_DDPG_FS_30.zip")
# model.learning_rate = 0.0003
# model.gamma = 0.99
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.05*np.ones(n_actions))
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.075 * np.ones(n_actions))
# model.action_noise = action_noise
trainer = Trainer(env)
trainer.retrain_rl(model, episodes=20000)