import train
import numpy as np
import os

"""
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

env_name = 'FetchPush-v1'
mode = 'classic'
episodes = 100
random_sample = False

data_savefile = './tmp/' + env_name + '/'

samples = 20
saved_stats = np.empty((episodes+1, 5, 0))
for i in range(samples):
    print('Starting run {} in {}'.format(i+1, env_name))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                data_savefile, expert_first=False, save_model=True)
    stats = np.array(stats)
    saved_stats = np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)

    if random_sample:
        sf = data_savefile + env_name + '-' + mode + '-random-multi.npy'
    else:
        sf = data_savefile + env_name + '-' + mode + '-multi.npy'
    np.save(sf, saved_stats)
