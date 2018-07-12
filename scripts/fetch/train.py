import gym
import gym.spaces
import pickle
import tensorflow as tf
import os
import random

from active_imitation.experts import RoboticEnv_Expert
from active_imitation.agents import GymRobotAgent
from active_imitation.utils import configure
from active_imitation.agents.mujoco_robot import DEFAULT_PARAMS
from active_imitation.experts import trained_models


"""
Use this script for training on the Fetch robot environments
through OpenAI Gym and Mujoco

Possible environments are:
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

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

def main(env_name, mode, episodes, random_sample, save_path, expert_first=False, save_model=True):
    """
    env_name - gym environment
    mode - learning type [pool, stream, classic]
    episodes - how many episodes to train the agent for
    save_path - where the model and tf loggin data should be saved to
    random_sample - If the label samples should be selected randomly, else actively
    expert_first - Should the expert have control for the first episode or no
    """

    seed = random.randint(0, 1e6)
    env = gym.make(env_name)
    env.seed(seed)
    
    if expert_first:
        mixing = 0.0
    
    # Need the spaces dimensions to initialize the NN agent    
    action_size = env.action_space.shape[0]
    observation_size = env.observation_space.spaces['observation'].shape[0]
    goal_size = env.observation_space.spaces['desired_goal'].shape[0]
    env_dims = {'observation':observation_size, 'goal':goal_size, 'action':action_size}
    
    # Change the dimensions of the nn layers
    params = DEFAULT_PARAMS
    params['layers'] = [256, 256, 256]
    params['dropout_rate'] = 0.00
    params['filepath'] = save_path
    param_mods = {'random_sample': random_sample}

    agent = GymRobotAgent(env_dims, **DEFAULT_PARAMS)
    expert = RoboticEnv_Expert(policy_files[env_name])
    
    learning_mode = configure.configure_robot(env, env_dims, agent, expert, mode,
                                              continuous=True, param_mods=param_mods)                        
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
    save_path = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchSlide-v1/Classic/'
    main(save_path)
    
    

