import time
import random
import csv
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from gym import spaces, ObservationWrapper, RewardWrapper, ActionWrapper


class Trainer:
    def __init__(self, env):
        self.env = env
        self.param = env.param

    def train_rl(self, models_to_train=40, episodes_per_model=100):
        # specify the RL algorithm to train (eg ACKTR, TRPO...)

        # Callback for saving the best agent during training
        eval_callback = EvalCallback(self.env, best_model_save_path='./logs/',
                                     log_path='./logs/', eval_freq=500,
                                     deterministic=True, render=False)

        model = PPO(MlpPolicy, self.env, verbose=1, learning_rate=0.0003, tensorboard_log="./logs/")
        start = time.time()

        for i in range(models_to_train):
            steps_per_model = episodes_per_model * self.param.steps_per_episode
            model.learn(total_timesteps=steps_per_model, callback=eval_callback)
            model.save("MODEL_" + str(i))

        end = time.time()
        print("time (min): ", (end - start) / 60)

    def retrain_rl(self, model, episodes):
        # Method for retraining a saved model for more timesteps
        start = time.time()

        # Callback for saving the best agent during training
        eval_callback = EvalCallback(self.env, best_model_save_path='./logs/',
                                     log_path='./logs/', eval_freq=500,
                                     deterministic=True, render=False)
        steps_per_model = episodes * self.param.steps_per_episode

        model.set_env(self.env)
        model.learn(total_timesteps=steps_per_model, callback=eval_callback)
        model.save("MODEL_RETRAINED")

        end = time.time()
        print("time (min): ", (end - start) / 60)


class Evaluate:
    def __init__(self, env, agent=None):
        self.env = env
        self.param = env.param
        self.agent = agent

    def generate_random_seeds(self, n, fname="test_set_seeds.csv"):
        seeds = [random.randint(0, 1e7) for i in range(n)]
        df = pd.DataFrame(seeds)
        df.to_csv(fname, index=False, header=False)

    def read_seeds(self, fname="test_set_seeds.csv"):
        file = open(fname)
        csv_file = csv.reader(file)
        seeds = []
        for row in csv_file:
            seeds.append(int(row[0]))
        self.seeds = seeds
        return seeds

    def min_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                if type(self.env.action_space) == gym.spaces.discrete.Discrete:
                    action = 0
                elif type(self.env.action_space) == gym.spaces.Box:
                    action = self.env.action_space.low
                    # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def max_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                if type(self.env.action_space) == gym.spaces.discrete.Discrete:
                    action = self.env.action_space.n - 1
                elif type(self.env.action_space) == gym.spaces.Box:
                    action = self.env.action_space.high
                # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def random_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = self.env.action_space.sample()
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        # TODO we should double check that sampling from the observation space is independent from
        # sampling in the environment which happens with fixed seed
        return np.mean(rewards)

    def RL_agent(self, seeds):
        rewards = []
        model = self.agent
        for seed in seeds:
            self.env.seed(seed)
            obs = self.env.reset()
            while not self.env.state.is_done():
                action, _states = model.predict(obs,deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def matching_agent(self, seeds):
        # This uses the expensive generator to poorly match the predicted generation.
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            obs = self.env.step([2,0])
            while not self.env.state.is_done():
                current_time = obs[0][0]
                current_generation_1 = obs[0][1]
                current_generation_2 = obs[0][2]
                forecasts = obs[0][3:]
                predicted_generation = forecasts[current_time]
                extra_generation = predicted_generation - current_generation_1 - current_generation_2
                action = [current_generation_1, current_generation_2 + extra_generation]
                obs = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def transformed_agent(self, seeds, H, transform):
        rewards = []
        model = self.agent
        for seed in seeds:
            self.env.seed(seed)
            obs = self.env.reset()
            while not self.env.state.is_done():
                obs = ObservationTransform(obs, H, transform)
                action, _states = model.predict(obs, deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

### The reason I am defining this ObservationMapping as a separate function and not a method is
### because we want the exact same function to be used when running the evaluation of the agent.
### Particularly important when submitting the agent to Rangl.

def ObservationTransform(obs, H, transform, steps_per_episode=int(96)):
    step_count, generator_1_level, generator_2_level = obs[:3]
    agent_prediction = np.array(obs[3:])  # since it was stored as a tuple
    # Initiate the padding values to 2 (the mean value), but note that this is a design choice
    # It might be better to initate them to -99999999 (a large number) if that means they interfere less
    # It might not matter
    agent_horizon_prediction = agent_prediction[-1] * np.ones(steps_per_episode)
    agent_horizon_prediction[:int(steps_per_episode - step_count)] = agent_prediction[int(step_count):]  # inclusive index
    agent_horizon_prediction = agent_horizon_prediction[:H]

    if transform == "Standard":
        pass
    if transform == "Zeroed":
        agent_horizon_prediction -= agent_prediction[step_count] * np.ones(H)
    if transform == "Deltas":
        # TODO: test this
        agent_horizon_prediction = np.concatenate(([agent_prediction[step_count]],
                                                   agent_horizon_prediction))
        agent_horizon_prediction = np.diff(agent_horizon_prediction)


    # print("Working: horizon obs = {}".format(obs))

    steps_to_peak = list(agent_prediction).index(max(agent_prediction)) - step_count
    obs = (step_count, steps_to_peak, generator_1_level, generator_2_level) + tuple(agent_horizon_prediction)  # repack

    return obs

class HorizonObservationWrapper(ObservationWrapper):

    def __init__(self, env, horizon_length, transform_name):

        super(HorizonObservationWrapper, self).__init__(env)

        self.H = horizon_length

        # Different transform methods
        transform_options = ["Standard", "Zeroed", "Deltas"]
        assert transform_name in transform_options, "Set a valid transform"
        self.transform = transform_name

        self.steps_per_episode = int(96)
        self.n_obs = len(ObservationTransform( tuple(np.ones(99,)), self.H , transform=self.transform))
        self.observation_space = self.get_observation_space()

    def get_observation_space(self):

        obs_low = np.full(self.n_obs, -1000, dtype=np.float32)  # last 96 entries of observation are the predictions
        obs_low[0] = -1  # first entry of obervation is the timestep
        obs_low[1] = 0.5  # min level of generator 1
        obs_low[2] = 0.5  # min level of generator 2
        obs_high = np.full(self.n_obs, 1000, dtype=np.float32)  # last 96 entries of observation are the predictions
        obs_high[0] = self.param.steps_per_episode  # first entry of obervation is the timestep
        obs_high[1] = 3  # max level of generator 1
        obs_high[2] = 2  # max level of generator 2
        result = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return result

    def observation(self, obs):

        # Apply the globally defined ObservationTransform transform to the observations
        obs = ObservationTransform(obs, self.H, transform=self.transform, steps_per_episode=self.steps_per_episode)

        return obs


class PhaseRewardWrapper(RewardWrapper):
    def __init__(self, env, phase="Full"):
        super(PhaseRewardWrapper, self).__init__(env)

        assert phase in ["Warmup", "Peak", "Full"], "Set valid phase."
        self.phase = phase

    def reward(self, rew):

        if self.phase=="Warmup" and self.env.state.step_count != 1:
            rew = 0

        if self.phase=="Peak" and self.env.state.step_count != 5:
            rew = 0

        return rew
