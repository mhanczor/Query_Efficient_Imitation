import gym
import gym.spaces
import pickle
import tensorflow as tf
import os

from active_imitation.experts import CartPole_iLQR, LunarLander_Expert, RoboticEnv_Expert
from active_imitation.experts import trained_models
from active_imitation.learners import DAgger

import numpy as np

env_name = 'CartPole-v1'
validation_runs = 1000


experts = {'CartPole-v1' : CartPole_iLQR,
           'LunarLander-v2' : LunarLander_Expert} 

prefix = os.path.dirname(trained_models.__file__)
policy_files = {'FetchReach-v1': os.path.join(prefix, 'FetchReach-v1/policy_best.pkl'),
                'FetchSlide-v1': os.path.join(prefix, 'FetchSlide-v1/policy_best.pkl'),
                'FetchPickAndPlace-v1': os.path.join(prefix, 'FetchPickAndPlace-v1/policy_best.pkl'),
                'FetchPush-v1': os.path.join(prefix, 'FetchPush-v1/policy_best.pkl')}

# I think what I can do is just load everything into DAgger and use the expert as 
# the agent in this case.  Then by just running validAgent I should get the 
# successes, and validation rewards I'd want

env = gym.make(env_name)

isFetch = env_name[:5] == 'Fetch'
if isFetch:
    agent = RoboticEnv_Expert(policy_files[env_name])
    continuous = True
else:
    continuous = False
    if env_name == 'CartPole-v1': 
        agent = CartPole_iLQR(env.unwrapped)
    if env_name == 'LunarLander-v2':
        agent = LunarLander_Expert(env.unwrapped)
        
        
learning_mode = DAgger(env, learner=agent, expert=agent, 
                agg_buffer=None, continuous=continuous)
                
valid_reward, valid_acc, avg_success = learning_mode.validateAgent(validation_runs)

print('Validation Reward: {} \n \
       Validation Accuracy: {} \n \
       Average Successes: {}'.format(valid_reward, valid_acc, avg_success))

savefile = './data/' + env_name + '-performance.txt'

with open(savefile, 'w') as f:
    f.write('Environment: {} \n'.format(env_name))
    f.write('# of Runs: {} \n'.format(validation_runs))
    f.write('Average Reward: {} \n'.format(valid_reward))
    f.write('Average Success Rate: {}\n'.format(avg_success))
                
                
