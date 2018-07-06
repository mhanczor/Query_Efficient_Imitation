import gym
import gym.spaces
import pickle

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
policy_file = '/tmp/openai-2018-07-03-17-00-07-524061/policy_best.pkl'

env = gym.make('FetchReach-v1')
state = env.reset()
o = state['observation']
ag = state['achieved_goal']
g = state['desired_goal']

with open(policy_file, 'rb') as f:
    policy = pickle.load(f)
    
policy_output = policy.get_actions(o, ag, g, compute_Q=True)
action, Q_val = policy_output

env.step(action)

# This should be everything that is needed to get a learner up to speed.


# How are we gonna handle multiple actions in an action space?
# In the actor crtitic approach, actions are output using a tanh nonlinearity
# and then scaled by a max value multiplier