import train
import numpy as np
import os

"""
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

env_name = 'FetchReach-v1'
mode = 'classic'
episodes = 100
random_sample = False

data_savefile = './tmp/trained_data/'
save_path = './tmp/' + env_name + '/'

samples = 100
saved_stats = np.empty((episodes+1, 4, samples))
for i in range(samples):
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                save_path, expert_first=False, save_model=True)
    stats = np.array(stats)
    saved_stats[:,:,i] = stats[:,:]

if not os.path.exists(data_savefile):
    os.makedirs(data_savefile)

np.save(data_savefile + env_name + '-multi.npy', saved_stats)