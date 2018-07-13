import gym
import gym.spaces
import pickle
import tensorflow as tf
import os
import random

from active_imitation.experts import CartPole_iLQR, LunarLander_Expert
from active_imitation.agents import GymAgent
from active_imitation.agents.classic_gym import DEFAULT_PARAMS
from active_imitation.utils import configure


"""
Use this script for training on the classic OpenAI gym environments

Possible environments are:
CartPole-v1
LunarLander-v2
"""

experts = {'CartPole-v1' : CartPole_iLQR,
           'LunarLander-v2' : LunarLander_Expert} 


#######
# Assorted training variables
mixing = 1.0
mixing_decay = 1.0
train_epochs = 10

#######

def main(env_name, mode, episodes, random_sample, save_path, expert_first=False, save_model=True, dropout=0.05):
    """
    env_name - gym environment [LunarLander-v2, CartPole-v1]
    mode - learning type [pool, stream, classic]
    save_path - where the model and tf loggin data should be saved to
    """
    seed = random.randint(0, 1e6)
    env = gym.make(env_name)
    env.seed(seed)

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
    
    if expert_first:
        mixing = 0.0
        mixing_decay = 1.0
    else:
        mixing = 1.0
        mixing_decay = 1.0
    
    param_mods = {'random_sample': random_sample, 'mixing':mixing}

    agent = GymAgent(env_dims, **DEFAULT_PARAMS)
    expert = experts[env_name](env.unwrapped)
    
    learning_mode = configure.configure_robot(env, env_dims, agent, expert, 
                                              mode, continuous=False, param_mods=param_mods)                        
    rewards, stats = learning_mode.train(episodes=episodes, 
                                        mixing_decay=mixing_decay,
                                        train_epochs=train_epochs,
                                        save_images=False,
                                        image_filepath=save_path+'images/')
    if save_model:
        agent.save_model()
    agent.sess.close()
    env.close()
    return rewards, stats


if __name__ == "__main__":
    path = './'
    main(save_path)
    
    

