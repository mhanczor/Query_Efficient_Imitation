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
episodes = 500 #50 for LL, 20 for CP
random_sample = True
dropout = 0.05
concrete = True
learning_rate = 1e-3
run_no = 'hetero_loss-0'
samples = 4

data_savepath = './tmp/' + env_name + '/'
saved_stats = None

for i in range(samples):
    
    data_savefile = data_savepath + env_name + '-' + mode
    if random_sample:
         data_savefile += '-random'
    if concrete:
        data_savefile += '-concrete'
    data_savefile += '-multi'
    if run_no != '': data_savefile += '-'+run_no
    data_savefile += '/'
    
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
    sf += '-multi'
    if run_no != '': sf += '-'+run_no
    sf += '.npy'
    
    np.save(sf, saved_stats)
    
    
    
    
    
    
    
    