#!/usr/bin/env python

import gym
from stable_baselines3 import PPO

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from util import Evaluate

env = gym.make("reference_environment:reference-environment-v0")

### Question is - Can this work with a Finite Horizon Linear Policy???

class AlexAgent:

    def __init__(self, H):

        self.H = H # Number of input variables - i.e. Horizon length
        self.A = 2 # Number of actions
        self.policy_coefficients = np.zeros((2, H))

    def padding(self, L):
        return np.concatenate((np.array(L), L[-1]*np.ones(self.H - len(L))))

    def policy(self, inputs):
        return self.policy_coefficients @ inputs

    def read_observation(self, obs):
        return obs[0], obs[1], obs[2], obs[3:]

    def make_prediction(self, obs):
        step, gen1, gen2, L = self.read_observation(obs)

        if len(L) < self.H:
            # Pad the observations with L(96) if the horizon exceeds L(96).
            L = self.padding(L)
        else:
            # Trim forecasts at the horizon.
            L = L[step:self.H+step]

        # Remove current generation level from forecasts
        # TODO: see how this affects the policy learning
        L = L - np.ones(self.H)*gen1 - np.ones(self.H)*gen2

        # TODO: try using the differences of this vector
        L = np.diff(np.concatenate(([0],L)))

        return list(self.policy(L))

    def predict(self, obs, deterministic=True):
        # Interface with Evaluate
        return self.make_prediction(obs), None

H=3
agent = AlexAgent(H=H)
evaluate = Evaluate(env, agent)

seeds = evaluate.read_seeds(fname="seeds_original.csv")
mean_reward = evaluate.RL_agent(seeds)


def linear_policy_optimisation(env, agent, p0):

    seeds = evaluate.read_seeds(fname="seeds_original.csv")

    def func(vec):
        agent.policy_coefficients = vec.reshape((2, agent.H))
        evaluate = Evaluate(env, agent)
        mean_reward = evaluate.RL_agent(seeds)
        print(vec)
        return -mean_reward

    result = minimize(func, p0.ravel(), method = 'BFGS', options={'gtol':1e-20, 'disp':True, 'maxiter':100, 'norm':2})

    return result

np.random.seed(None)
result = linear_policy_optimisation(env, agent, abs(np.random.randn(2, H)))


agent = AlexAgent(H=120)
L = np.ones(100)
L[-1] = 0.5
assert len(agent.padding(np.ones(100))) == 120
assert np.all(agent.padding(L)[-20:] == 0.5)

env.reset()
agent = AlexAgent(H=5)
obs, _, _, _ = env.step([2,0])
action = agent.predict(obs)
assert len(action) == 2



policy_linear_result_H_5 = np.array([  0.11207772 ,  3.75299497 ,  2.97851625 ,  3.28880267 ,  7.58495966 ,
  -8.12853111 ,  -9.97521518 ,  -12.4143787 ,  -10.18381189 ,  -10.8786626 ])