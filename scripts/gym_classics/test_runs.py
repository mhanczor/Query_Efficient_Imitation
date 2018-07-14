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
episodes = 1500 #50 for LL, 20 for CP
random_sample = False

data_savefile = './tmp/' + env_name + '/'

samples = 5

#### DROPOUT 0.05 QBC-KL
# special_tag = [['DO-0-1', 0.1], ['DO-0-2', 0.2], ['DO-0-3', 0.3], ['DO-0-05', 0.05], ['DO-0-02', 0.02], ['DO-0-005', 0.005]]

special_tag = [['Utility_Monitor_1500_Steps', 0.05]]

for j in range(len(special_tag)):
    saved_stats = np.empty((episodes+1, 6, 0))
    for i in range(samples):
        data_path = data_savefile + special_tag[j][0] + '/'
        print('Starting run {} with dropout {}'.format(i+1, special_tag[j][0]))
        rewards, stats = train.main(env_name, mode, episodes, random_sample, 
                                    data_path + str(i) + '/', expert_first=expert_first, 
                                    save_model=save_model, dropout=special_tag[j][1])
        stats = np.array(stats)
        saved_stats = np.append(saved_stats, stats[:,:, None], axis=2)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        sf = data_path + env_name + '-' + mode + '-' + special_tag[j][0]+'.npy'
        np.save(sf, saved_stats)