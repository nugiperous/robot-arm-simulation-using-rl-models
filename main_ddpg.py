
import numpy as np
from ddpg_torch import DDPGAgent
#from ddpg_example import DDPGAgent
from RobotArmEnv import RobotArmEnv
from utils import plot_learning_curve

if __name__ == '__main__':
    env = RobotArmEnv()
    env.reset()
    env.define_goal(2)
    agent = DDPGAgent(alpha=0.0001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=54, fc2_dims=54, 
                    n_actions=env.action_space.shape[0])
    n_games = 500
    filename = 'RobotArm_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    
    for i in range(n_games):
        observation = env.reset()
        #observation = observation['observation']
        done = False
        score = 0
        agent.noise.reset()
        max_ittr = 2000
        ittr = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #observation_ = observation_['observation']
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
            ittr = ittr + 1
            
            if ittr >= max_ittr:
                break  # break out of the while loop

            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)



