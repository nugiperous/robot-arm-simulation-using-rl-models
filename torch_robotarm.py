"""
Created on Thu May  4 19:42:15 2023

@author: froot
"""

import gymnasium as gym
import numpy as np
from ddpg_torch import DDPGAgent
from stable_baselines3 import DDPG, HerReplayBuffer
from utils import plot_learning_curve
import panda_gym
import os


env = gym.make('PandaReach-v3')
observation, info = env.reset()
n_actions = env.action_space.shape[0]

agent = DDPGAgent(alpha=0.0001, beta=0.001, 
                input_dims=(8,8,8), tau=0.001,
                batch_size=64, fc1_dims=400, fc2_dims=300, 
                n_actions=n_actions)
n_games = 1000
filename = 'PandaReach_alpha_' + str(agent.alpha) + '_beta_' + \
            str(agent.beta) + '_' + str(n_games) + '_games'
figure_file = os.path.join('plots', filename + '.png')

best_score = env.reward_range[0]
score_history = []
for i in range(n_games):
    observation, info = env.reset()
    done = False
    score = 0
    agent.noise.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info, _ = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode ', i, 'score %.1f' % score,
            'average score %.1f' % avg_score)
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, figure_file)



