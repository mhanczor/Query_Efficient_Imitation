import gym
import gym.spaces
import pickle
import tensorflow as tf
import os

from active_imitation.experts import RoboticEnv_Expert
from active_imitation.agents import GymRobotAgent
from active_imitation.utils import configure
from active_imitation.agents.mujoco_robot import DEFAULT_PARAMS

"""
Use this script for training on the Fetch robot environments
through OpenAI Gym and Mujoco

Possible environments are:
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1

May extend this for HandManipulation tasks
"""

prefix = '/home/hades/Research/Active_Imitation/active_imitation/experts/trained_models/'
policy_files = {'FetchReach-v1': os.path.join(prefix, 'FetchReach-v1/policy_best.pkl'),
                'FetchSlide-v1': os.path.join(prefix, 'FetchSlide-v1/policy_best.pkl'),
                'FetchPickAndPlace-v1': os.path.join(prefix, 'FetchPickAndPlace-v1/policy_best.pkl'),
                'FetchPush-v1': os.path.join(prefix, 'FetchPush-v1/policy_best.pkl')}

#######
# Assorted training variables
mixing = 1.0
mixing_decay = 1.0
random_sample = False
episodes = 20
train_epochs = 10
filepath = '~/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/First_Test/'
#######

def main(save_path):
    mode = 'pool'
    env_name = 'FetchReach-v1'
    env = gym.make(env_name)
    sess = tf.Session()
    
    # Need the spaces dimensions to initialize the NN agent    
    action_size = env.action_space.shape[0]
    observation_size = env.observation_space.spaces['observation'].shape[0]
    goal_size = env.observation_space.spaces['desired_goal'].shape[0]
    env_dims = {'observation':observation_size, 'goal':goal_size, 'action':action_size}
    
    # Change the dimensions of the nn layers
    params = DEFAULT_PARAMS
    params['layers'] = [256, 256, 256]
    params['dropout_rate'] = 0.1

    agent = GymRobotAgent(sess, env_dims, **DEFAULT_PARAMS)
    expert = RoboticEnv_Expert(policy_files[env_name])
    
    learning_mode = configure.configure_robot(env, env_dims, agent, expert, mode)                        
    rewards, stats = learning_mode.train(episodes=episodes, 
                                        mixing_decay=mixing_decay,
                                        train_epochs=train_epochs,
                                        save_images=False,
                                        image_filepath=filepath+'images/')
    # Try saving the agent
    with open(save_path, 'wb') as f:
        import ipdb; ipdb.set_trace()
        pickle.dump(agent, f)
        print('Saved file to {}'.format(save_path))
    
    print('Closing TensorFlow Session')
    sess.close()
    return

def play(save_path):
    # sess = tf.Session()
    import ipdb; ipdb.set_trace()
    print('Loading Saved Policy')
    with open(save_path, 'rb') as f:
        policy = pickle.load(f)
    
    env = gym.make('FetchReach-v1')
    state = env.reset()
    action = policy.samplePolicy(state, True)
    print(action)
    return

if __name__ == "__main__":
    save_path = './test.pkl'
    main(save_path)
    play(save_path)
