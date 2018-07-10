import gym
import gym.spaces
import pickle
import tensorflow as tf
import os

from active_imitation.experts import RoboticEnv_Expert
from active_imitation.agents import GymRobotAgent
from active_imitation.utils import configure
from active_imitation.agents.mujoco_robot import DEFAULT_PARAMS


def play(save_path):
    env_name = 'FetchReach-v1'
    env = gym.make(env_name)
    
    # Need the spaces dimensions to initialize the NN agent    
    action_size = env.action_space.shape[0]
    observation_size = env.observation_space.spaces['observation'].shape[0]
    goal_size = env.observation_space.spaces['desired_goal'].shape[0]
    env_dims = {'observation':observation_size, 'goal':goal_size, 'action':action_size}
    
    # Change the dimensions of the nn layers
    params = DEFAULT_PARAMS
    params['layers'] = [256, 256, 256]
    params['dropout_rate'] = 0.1
    params['filepath'] = save_path
    
    policy = GymRobotAgent(env_dims, load=True, **DEFAULT_PARAMS)

    import ipdb; ipdb.set_trace()
    for i in range(10):
        done = False
        state = env.reset()
        env.render()
        while not done:
            action = policy.sampleAction(state, True).flatten()
            state, reward, done, _ = env.step(action)
            env.render()
    return
    
def main():
    save_path = ''
    play(save_path)

if __name__ == "__main__":
    main()