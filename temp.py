# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:52:24 2023

@author: froot
"""

import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3 .common.noise import NormalActionNoise
#from sb3_contrib.common.wrappers import TimeFeatureWrapper



env = gym.make('PandaReach-v3')

rb_kwargs = {'online_sampling' : True,
             'goal_selection_strategy' : 'future',
             'n_sampled_goal' : 4}

policy_kwargs = {'net_arch' : [512, 512, 512], 
                 'n_critics' : 2}

n_actions = env.action_space.shape[0]
noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

env = gym.make("PandaReach-v3")
#env = TimeFeatureWrapper(env)

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
             gamma = 0.95, batch_size= 2048, buffer_size=100000, replay_buffer_kwargs = rb_kwargs,
             learning_rate = 1e-3, action_noise = noise, policy_kwargs = policy_kwargs)
model.learn(1e6)
model.save('pick_place/model')