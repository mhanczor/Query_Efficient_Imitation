import numpy as np
import gym
import tensorflow as tf

"""
Run multiple training cycles to gain confidence bounds on the learning 
characteristics of different approaches

Unless otherwise specified the plots will use standard error of the mean as 
the confidence bounds
"""


#############
# Parmeters #
#############

trials = 100
filename = ''

episodes = 200
env_name = 'LunarLander-v2' # 'CartPole-v1'
criteria_threshold = 0.5

mixing = 1.0 # Initial mixing value, geq 1.0 results in no expert controlled runs
mixing_decay = 1.0 # Rate of change of the mixing value per episode
train_epochs = 15
dropout_rate = 0.01
random_sample = False # Radoom_Sample true overrides the specified selection criteria

savepath = '~/Research/experiments/confidence_runs/'
filepath = savepath + filename
env = gym.make(env_name)
sess = tf.Session()
agent = GymAgent  (sess, env, lr = 0.001, dropout_rate=dropout_rate, filepath=filepath)





