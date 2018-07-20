import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchReach Expert Perfomance
expert_valid = -1.709
expert_success = 1.0

# Stats are [0:Episode, 1:Total Expert Samples, 2:Expert Samples/Episode, 
#           3:Validation Reward, 4:Average Successes, 5:Uncertainty of Selected Sample, 
#           6:Final Training Loss]

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/'

DAgger_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-classic-concrete-multi-baseline.npy')

Random_fp = os.path.join(prefix, '0715/FetchReach-v1-pool-random-concrete-multi-long/FetchReach-v1-pool-random-concrete-multi-long.npy')
Concrete_fp = os.path.join(prefix, '0715/FetchReach-v1-pool-concrete-multi-long/FetchReach-v1-pool-concrete-multi-long.npy')

rand_basic = os.path.join(prefix, '0719/FetchReach-v1-pool-random-multi-basic.npy')
rand_basic_1e5 = os.path.join(prefix, '0719/FetchReach-v1-pool-random-multi-basic_1e5.npy')
dw_05 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_0-5.npy')
dw_10 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_1-0.npy')
dw_5e4 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_5e4.npy')

filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp, 'Concrete':Concrete_fp,
'Rand Basic':rand_basic, 'Rand Basic 1e5':rand_basic_1e5, 'DW 0.5':dw_05, 'DW 1.0':dw_10, 'DW 5e-4':dw_5e4}
s_val = 13
smoothing = {'DAgger':0, 'Random':s_val, 'Concrete':s_val}
# filepaths = {'Concrete':concrete_fp}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot_labels = ['Expert_Samples', 'Validation Reward', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, 1000), ylims=None, smoothing=None)
plot_labels = ['Expert_Samples', 'Success Rate', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0, 1000), ylims=None, smoothing=None)

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, 1000), ylims=None, smoothing=None)

s_val = 15
# Only plot Random and Concrete here? Why?
smoothing = {'DAgger':s_val, 'Random':s_val, 'Concrete':s_val}
plot_labels = ['Expert_Samples', 'Training Loss', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=None, ylims=None, smoothing=None)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=None, ylims=None, smoothing=none)
# Old file paths
# 
# Pre-concrete dropout baselines
# DAgger_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-classic-multi.npy')
# Random_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-pool-random-multi.npy')



# test_fp = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/0714/FetchReach-v1-pool-multi.npy'
# concrete_fp = os.path.join(prefix, '0714/FetchReach-v1-pool-concrete-multi/FetchReach-v1-pool-concrete-multi.npy')
# concrete_2 = os.path.join(prefix, '0715/FetchReach-v1-pool-concrete-multi-0/FetchReach-v1-pool-concrete-multi.npy')
# concrete_random = os.path.join(prefix, '0715/FetchReach-v1-pool-random-concrete-multi/FetchReach-v1-pool-random-concrete-multi.npy')
