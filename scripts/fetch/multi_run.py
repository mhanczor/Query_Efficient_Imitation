import numpy as np
import os
from scripts import train
import time

"""
FetchReach-v1
FetchSlide-v1
FetchPickAndPlace-v1
FetchPush-v1
"""

env_name = 'FetchPush-v1'
mode = 'pool'
expert_first = True
save_model = True
episodes = 3
random_sample = False
dropout = 0.05
concrete = True
learning_rate = 1e-4
run_no = 'dw_test'
samples = 2
ls = 1e-4
train_epochs = 10
density = 1.0

data_savepath = './tmp/' + env_name + '/'
saved_stats = None

t0 = time.time()
past_time = 0.0
tot_time = 0.0
for i in range(samples):
    ep_start = time.time()
    data_savefile = data_savepath + env_name + '-' + mode
    
    if random_sample:
         data_savefile += '-random'
    if concrete:
        data_savefile += '-concrete'
    data_savefile += '-multi'
    if run_no != '': data_savefile += '-'+run_no
    data_savefile += '/'
        
    
    print('\n \n Starting run {} in {}\n \
Last Episode Length: {} Total Training Time: {} \n'.format(i+1, env_name, round(past_time, 3), round(tot_time, 3)))
    rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                data_savefile + str(i) + '/', expert_first=expert_first, 
                                save_model=save_model, dropout=dropout, concrete=concrete,
                                lr=learning_rate, ls=ls, train_epochs=train_epochs, density=density)
    stats = np.array(stats)
    saved_stats = np.atleast_3d(stats) if saved_stats is None else np.append(saved_stats, stats[:,:, None], axis=2)

    if not os.path.exists(data_savefile):
        os.makedirs(data_savefile)

    sf = data_savefile + env_name + '-' + mode
    if random_sample:
        sf += '-random'
    if concrete:
        sf += '-concrete'
    sf += '-multi'
    if run_no != '': sf += '-'+run_no
    sf += '.npy'
        
    np.save(sf, saved_stats)
    past_time = time.time() - ep_start
    tot_time = time.time() - t0

print('Training {} samples for {} episodes took {} minutes'.format(samples, episodes, round(tot_time/60., 3)))
