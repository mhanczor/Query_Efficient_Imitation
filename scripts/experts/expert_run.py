import gym
import gym.spaces
import pickle
import tensorflow as tf
import os

from active_imitation.experts import CartPole_iLQR, LunarLander_Expert, RoboticEnv_Expert, SpaceInvadersExpert
from active_imitation.experts import trained_models
from active_imitation.learners import DAgger

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

import numpy as np

env_name = 'SpaceInvadersNoFrameskip-v0'
validation_runs = 30

prefix = os.path.dirname(trained_models.__file__)
policy_files = {'FetchReach-v1': os.path.join(prefix, 'FetchReach-v1/policy_best.pkl'),
                'FetchSlide-v1': os.path.join(prefix, 'FetchSlide-v1/policy_best.pkl'),
                'FetchPickAndPlace-v1': os.path.join(prefix, 'FetchPickAndPlace-v1/policy_best.pkl'),
                'FetchPush-v1': os.path.join(prefix, 'FetchPush-v1/policy_best.pkl')}

# I think what I can do is just load everything into DAgger and use the expert as 
# the agent in this case.  Then by just running validAgent I should get the 
# successes, and validation rewards I'd want


kwargs = {}
isFetch = env_name[:5] == 'Fetch'
if isFetch:
    env = gym.make(env_name)
    agent = RoboticEnv_Expert(policy_files[env_name])
    continuous = True
else:
    continuous = False
    if env_name == 'CartPole-v1': 
        env = gym.make(env_name)
        agent = CartPole_iLQR(env.unwrapped)
    if env_name == 'LunarLander-v2':
        env = gym.make(env_name)
        agent = LunarLander_Expert(env.unwrapped)
    if env_name == 'SpaceInvadersNoFrameskip-v0':
        wrapper_kwargs = {'episode_life':False}
        env = VecFrameStack(make_atari_env(env_name, 1, 0, wrapper_kwargs=wrapper_kwargs), 4)
        kwargs['isSpace'] = True
        env_dims = {'observation':env.observation_space, 'action':env.action_space}
        agent = SpaceInvadersExpert(env_dims)
        
        
learning_mode = DAgger(env, learner=agent, expert=agent, 
                agg_buffer=None, continuous=continuous, **kwargs)
                
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
                
                
