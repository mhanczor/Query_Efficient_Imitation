import gym
import gym.spaces
import pickle
import tensorflow as tf
import os
import random

from active_imitation.experts import CartPole_iLQR, LunarLander_Expert, RoboticEnv_Expert, SpaceInvadersExpert
from active_imitation.agents import GymAgent, GymRobotAgent, AtariGymAgent

from active_imitation.utils import configure
from active_imitation.experts import trained_models

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

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

def main(env_name, mode, episodes, random_sample, save_path, concrete=False, expert_first=False, 
            save_model=True, dropout=0.05, lr=0.001, ls=5e-7, train_epochs=10, density=0.0, 
            hetero_loss=False, budget=1):
    """
    env_name - gym environment [LunarLander-v2, CartPole-v1]
    mode - learning type [pool, stream, classic]
    save_path - where the model and tf loggin data should be saved to
    """
    seed = random.randint(0, 1e6)
    isSpace = env_name[:5] == 'Space'
    if isSpace:
        wrapper_kwargs = {'episode_life':False}
        env = VecFrameStack(make_atari_env(env_name, 1, 0, wrapper_kwargs=wrapper_kwargs), 4)
    else:
        env = gym.make(env_name)
        env.seed(seed)
    
    isFetch = env_name[:5] == 'Fetch'
    if isFetch: # That's so fetch
        from active_imitation.agents.mujoco_robot import DEFAULT_PARAMS
        action_size = env.action_space.shape[0]
        observation_size = env.observation_space.spaces['observation'].shape
        goal_size = env.observation_space.spaces['desired_goal'].shape[0]
        env_dims = {'observation':observation_size, 'goal':goal_size, 'action':action_size}
    elif isSpace:
        from active_imitation.agents.classic_gym import DEFAULT_PARAMS
        action_size = 1
        action_space = env.action_space.n
        observation_size = env.observation_space.shape
        env_dims = {'observation':observation_size, 'action':action_size, 'action_space':action_space}
    else:
        from active_imitation.agents.classic_gym import DEFAULT_PARAMS
        # Need the spaces dimensions to initialize the NN agent    
        action_size = 1 # Single, discrete actions
        action_space = env.action_space.n
        observation_size = env.observation_space.shape
        env_dims = {'observation':observation_size, 'action':action_size, 'action_space':action_space}
    
    # Change the dimensions of the nn layers
    params = DEFAULT_PARAMS
    
    # params['layers'] = [64, 64, 64]
    params['dropout_rate'] = dropout #[0.05, 0.1, 0.15, 0.2]
    params['filepath'] = save_path
    params['lr'] = lr
    params['hetero_loss'] = hetero_loss
    if isFetch or isSpace:
        params['layers'] = [256, 256, 256] #[512, 512, 512] #
        params['concrete'] = concrete
        params['ls'] = ls
    else:
        params['layers'] = [16, 16, 16]
        params['concrete'] = concrete
    
    if expert_first:
        mixing = 0.0
        mixing_decay = 1.0
    else:
        mixing = 1.0
        mixing_decay = 1.0
    
    param_mods = {'random_sample': random_sample, 'mixing':mixing, 'density_weight':density, 'budget':budget}

    if isFetch:
        agent = GymRobotAgent(env_dims, **params)
        expert = RoboticEnv_Expert(policy_files[env_name])
        continuous = True
    elif isSpace:
        expert = SpaceInvadersExpert({'observation':env.observation_space, 'action':env.action_space})
        agent = AtariGymAgent(env_dims, **params)
        continuous = False
        param_mods['isSpace'] = True
    else:
        agent = GymAgent(env_dims, **params)
        expert = experts[env_name](env.unwrapped)
        continuous = False
    
    learning_mode = configure.configure_robot(env, env_dims, agent, expert, 
                                              mode, continuous=continuous, 
                                              concrete=concrete, param_mods=param_mods)  

    ## Save the training parameters
    # learning rate, dropout, isconcrete, iscontinuout, env_name, mode, 
    parameter_savefile = os.path.join(save_path, 'parameters.txt')
    with open(parameter_savefile, 'w') as f:
        f.write('Environment Name: {} \n'.format(env_name))
        f.write('Learning Mode: {} \n'.format(mode))
        f.write('# of Episodes: {} \n'.format(episodes))
        f.write('Learning Rate:{} \n'.format(lr))
        f.write('Concrete Length Scale: {} \n'.format(ls))
        f.write('Training Epochs: {}\n'.format(train_epochs))
        f.write('Continuous: {}\n'.format(continuous))
        f.write('Concrete: {}\n'.format(concrete))
        f.write('Random Sample: {}\n'.format(random_sample))
        f.write('Mixing: {}\n'.format(mixing))
        f.write('Mixing Decay: {}\n'.format(mixing_decay))
        f.write('Density Weighting: {}\n'.format(density))
        f.write('Budget: {}\n'.format(budget))
        for label, value in params.items():
            f.write('{}: {}\n'.format(label, value))
        f.write('Random Seed: {}\n'.format(seed))
    
    if isSpace:
        save_rate = 5000
        valid_runs = 1
    elif isFetch:
        save_rate = 100
        valid_runs = 5
    else:
        save_rate = 100
        valid_runs = 5
    rewards, stats = learning_mode.train(episodes=episodes, 
                                        mixing_decay=mixing_decay,
                                        train_epochs=train_epochs,
                                        save_images=False,
                                        image_filepath=save_path+'images/',
                                        save_rate=save_rate,
                                        valid_runs=valid_runs,)
    if save_model:
        agent.save_model()
    
    
    if isSpace:
        expert.close()
        
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
    
    

