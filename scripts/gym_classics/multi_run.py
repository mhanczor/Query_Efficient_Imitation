import numpy as np
import os
from scripts import train

"""
CartPole-v1
LunarLander-v2
"""

env_name = 'LunarLander-v2'#'CartPole-v1'
mode = 'pool'
expert_first = False
save_model = True
episodes = 1000 #50 for LL, 20 for CP
random_sample = True

data_savefile = './tmp/' + env_name + '/'

samples = 30
saved_stats = None

for i in range(samples):
    print('Starting run {} in {}'.format(i+1, env_name))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                data_savefile, expert_first=expert_first, save_model=save_model)
    stats = np.array(stats)
    saved_stats = np.atleast_3d(stats) if saved_stats is None else np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)
        
    if random_sample:
        sf = data_savefile + env_name + '-' + mode + '-random-multi.npy'
    else:
        sf = data_savefile + env_name + '-' + mode + '-multi.npy'
    np.save(sf, saved_stats)