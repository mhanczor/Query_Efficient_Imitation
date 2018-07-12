import train
import numpy as np
import os

"""
CartPole-v1
LunarLander-v2
"""

env_name = 'LunarLander-v2'#'CartPole-v1'
mode = 'classic'
episodes = 50 #50 for LL, 20 for CP
random_sample = False

data_savefile = './tmp/' + env_name + '/'

samples = 1
saved_stats = np.empty((episodes+1, 4, 0))
for i in range(samples):
    print('Starting run {}'.format(i+1))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                data_savefile, expert_first=False, save_model=False)
    stats = np.array(stats)
    saved_stats = np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)

    np.save(data_savefile + env_name + '-' + mode + '-multi.npy', saved_stats)