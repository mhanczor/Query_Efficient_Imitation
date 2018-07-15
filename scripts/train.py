import gym
import gym.spaces
import pickle
import tensorflow as tf
import os
import random

from active_imitation.experts import CartPole_iLQR, LunarLander_Expert, RoboticEnv_Expert
from active_imitation.agents import GymAgent, GymRobotAgent

from active_imitation.utils import configure
from active_imitation.experts import trained_models

"""
Use this script for training on the classic OpenAI gym environments

Possible environments are:
CartPole-v1
LunarLander-v2

Use this script for training on the Fetch robot environments
through OpenAI Gym and Mujoco

Possible environments are:
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

experts = {'CartPole-v1' : CartPole_iLQR,
           'LunarLander-v2' : LunarLander_Expert} 

prefix = os.path.dirname(trained_models.__file__)
policy_files = {'FetchReach-v1': os.path.join(prefix, 'FetchReach-v1/policy_best.pkl'),
                'FetchSlide-v1': os.path.join(prefix, 'FetchSlide-v1/policy_best.pkl'),
                'FetchPickAndPlace-v1': os.path.join(prefix, 'FetchPickAndPlace-v1/policy_best.pkl'),
                'FetchPush-v1': os.path.join(prefix, 'FetchPush-v1/policy_best.pkl')}

#######
# Assorted training variables
mixing = 1.0
mixing_decay = 1.0
train_epochs = 10

#######

def main(env_name, mode, episodes, random_sample, save_path, expert_first=False, save_model=True, dropout=0.05, concrete=False):
    """
    env_name - gym environment [LunarLander-v2, CartPole-v1]
    mode - learning type [pool, stream, classic]
    save_path - where the model and tf loggin data should be saved to
    """
    seed = random.randint(0, 1e6)
    env = gym.make(env_name)
    env.seed(seed)
    
    isFetch = env_name[:5] == 'Fetch'

    if isFetch: # That's so fetch
        from active_imitation.agents.mujoco_robot import DEFAULT_PARAMS
        action_size = env.action_space.shape[0]
        observation_size = env.observation_space.spaces['observation'].shape[0]
        goal_size = env.observation_space.spaces['desired_goal'].shape[0]
        env_dims = {'observation':observation_size, 'goal':goal_size, 'action':action_size}
    else:
        from active_imitation.agents.classic_gym import DEFAULT_PARAMS
        # Need the spaces dimensions to initialize the NN agent    
        action_size = 1 # Single, discrete actions
        action_space = env.action_space.n
        observation_size = env.observation_space.shape[0]
        env_dims = {'observation':observation_size, 'action':action_size, 'action_space':action_space}
    
    # Change the dimensions of the nn layers
    params = DEFAULT_PARAMS
    
    # params['layers'] = [64, 64, 64]
    params['dropout_rate'] = dropout #[0.05, 0.1, 0.15, 0.2]
    params['filepath'] = save_path
    if isFetch:
        params['layers'] = [256, 256, 256]
        params['concrete'] = concrete
    
    if expert_first:
        mixing = 0.0
        mixing_decay = 1.0
    else:
        mixing = 1.0
        mixing_decay = 1.0
    
    param_mods = {'random_sample': random_sample, 'mixing':mixing}

    if isFetch:
        agent = GymRobotAgent(env_dims, **params)
        expert = RoboticEnv_Expert(policy_files[env_name])
        continuous = True
    else:
        agent = GymAgent(env_dims, **params)
        expert = experts[env_name](env.unwrapped)
        continuous = False
    
    learning_mode = configure.configure_robot(env, env_dims, agent, expert, 
                                              mode, continuous=continuous, 
                                              concrete=concrete, param_mods=param_mods)  
                                                                    
    rewards, stats = learning_mode.train(episodes=episodes, 
                                        mixing_decay=mixing_decay,
                                        train_epochs=train_epochs,
                                        save_images=False,
                                        image_filepath=save_path+'images/')
    if save_model:
        agent.save_model()
    agent.sess.close()
    env.close()
    tf.reset_default_graph()
    return rewards, stats


if __name__ == "__main__":
    params = {'env_name':'LunarLander-v2',#'CartPole-v1'
              'mode':'pool',
              'expert_first':False,
              'save_model':True,
              'episodes':300,
              'random_sample':False,
              'save_path':'./tmp/LL_Test/'}
    params['dropout'] = 0.05
    main(**params)
    
    

