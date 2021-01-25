#!/usr/bin/env python
import gym
from stable_baselines3 import PPO
from pathlib import Path
from util import Evaluate
import numpy as np

env = gym.make("reference_environment:reference-environment-v0")

# agent = PPO.load("MODEL_0.zip")
from stable_baselines3 import DDPG

name_prefix = "models/model_DDPG_7_"
results = []
for steps in np.arange(400000,500000,1000):
    model_name = name_prefix + str(steps) + "_steps"

    agent = DDPG.load(model_name)

    evaluate = Evaluate(env, agent)
    seeds = evaluate.read_seeds(fname="seeds_original.csv")
    mean_reward = evaluate.transformed_agent(seeds, H=7,transform="Standard")

    print('Model number: ', str(model_name))
    print('Mean reward:', mean_reward)
    results.append((model_name, mean_reward))


