import numpy as np
import os
from scripts import train

"""
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

env_name = 'FetchReach-v1'
mode = 'pool'
expert_first = True
save_model = True
episodes = 1000
random_sample = False
dropout = 0.05
concrete = True
learning_rate = 5e-4
run_no = 'long'
samples = 10


data_savepath = './tmp/' + env_name + '/'
saved_stats = None

for i in range(samples):
    
    data_savefile = data_savepath + env_name + '-' + mode
    
    if random_sample:
         data_savefile += '-random'
    if concrete:
        data_savefile += '-concrete'
    data_savefile += '-multi-' + run_no+ '/'
        
    
    print('Starting run {} in {}'.format(i+1, env_name))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                data_savefile + str(i) + '/', expert_first=expert_first, 
                                save_model=save_model, dropout=dropout, concrete=concrete,
                                lr=learning_rate)
    stats = np.array(stats)
    saved_stats = np.atleast_3d(stats) if saved_stats is None else np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)

    sf = data_savefile + env_name + '-' + mode
    if random_sample:
        sf += '-random'
    if concrete:
        sf += '-concrete'
    sf += '-multi-' + run_no + '.npy'
        
    np.save(sf, saved_stats)
