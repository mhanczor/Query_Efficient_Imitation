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
episodes = 2
random_sample = False

data_savefile = './tmp/trained_data/'
save_path = './tmp/' + env_name + '/'

samples = 2
saved_stats = np.empty((episodes+1, 5, 0))
for i in range(samples):
    print('Starting run {}'.format(i+1))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                save_path, expert_first=False, save_model=True)
    stats = np.array(stats)
    saved_stats = np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)

    np.save(data_savefile + env_name + '-' + mode + '-multi.npy', saved_stats)
