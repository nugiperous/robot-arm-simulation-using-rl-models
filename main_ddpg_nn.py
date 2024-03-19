import numpy as np
from ddpg_torch import DDPGAgent
#from ddpg_example import DDPGAgent
from RobotArmEnv import RobotArmEnv
from utils import plot_learning_curve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = RobotArmEnv()
    env.reset()
    env.define_goal(2)
    fc1_list = [16,54,128]
    fc2_list = [16,54,128]
    total_score_history = []
    for fc1, fc2 in zip(fc1_list, fc2_list):
        print(f'FC1: {fc1} FC2: {fc2}')
        agent = DDPGAgent(alpha=.0001, beta=.001, 
                        input_dims=env.observation_space.shape, tau=0.001,
                        batch_size=64, fc1_dims=fc1, fc2_dims=fc2, 
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
            max_ittr = 4000
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
            
        total_score_history.append(score_history)
            
    #x = [i+1 for i in range(n_games)]
    #plot_learning_curve(x, score_history, figure_file)
    plt.figure()
    for i in range(len(fc1_list)):
        plt.plot(range(n_games), total_score_history[i], label = 'fc1_'+str(fc1_list[i])+'_fc2_'+str(fc2_list[i]))
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Loss Function of Total Reward for Each Episode')
    plt.legend()
    plt.show()

