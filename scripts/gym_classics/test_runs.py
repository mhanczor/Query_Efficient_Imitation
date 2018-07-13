import train
import numpy as np
import os

"""
CartPole-v1
LunarLander-v2
"""

env_name = 'LunarLander-v2'#'CartPole-v1'
mode = 'pool'
expert_first = False
save_model = True
episodes = 300 #50 for LL, 20 for CP
random_sample = False

data_savefile = './tmp/' + env_name + '/'

samples = 4
saved_stats = np.empty((episodes+1, 5, 0))

#### DROPOUT 0.05 QBC-KL
special_tag = [['DO-0-1', 0.1], ['DO-0-2', 0.2], ['DO-0-3', 0.3], ['DO-0-05', 0.05], ['DO-0-02', 0.02], ['DO-0-005', 0.005]]

for j in range(len(special_tag)):
    for i in range(samples):
        print('Starting run {} with dropout {}'.format(i+1, special_tag[j][0]))
        rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                    data_savefile, expert_first=expert_first, 
                                    save_model=save_model, dropout=special_tag[j][1])
        stats = np.array(stats)
        saved_stats = np.append(saved_stats, stats[:,:, None], axis=2)

        if not os.path.exists(data_savefile):
            os.makedirs(data_savefile)
            
        sf = data_savefile + env_name + '-' + mode + '-' + special_tag[j][0]+'.npy'
        np.save(sf, saved_stats)