import active_imitation.utils.plot as plot
import numpy as np
import os

#SpaceInvaders Expert Perfomance
expert_valid = 2486.17

xlim = 90000

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/SpaceInvaders-v0/'
DAgger_fp = os.path.join(prefix, '0803/SpaceInvadersNoFrameskip-v0-classic-multi-long_test_1.npy')
# Random_fp = os.path.join(prefix, '0724/SpaceInvadersNoFrameskip-v0-pool-random-multi-1ep_b10.npy')
Random_fp = os.path.join(prefix, '0803/SpaceInvadersNoFrameskip-v0-pool-random-multi-long_b30_test.npy')
JSD_fp = os.path.join(prefix, '0803/SpaceInvadersNoFrameskip-v0-pool-multi-long_b30_test.npy')


filepaths = {'DAgger':DAgger_fp,'Uniform':Random_fp, 'QBC':JSD_fp}
s_val = 111
smoothing = {'DAgger':1001, 'Uniform':s_val, 'QBC':s_val}

interpolate =['DAgger']

# filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp, 'QBC-JSB 0.00':jsd_0, 'QBC-JSB 0.00-1':jsd_1}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot_labels = ['Expert_Samples', 'Episode Reward', 'SpaceInvaders-v0']
plot.plotData(data, plot_labels, expert=None, data_axis=3, xlims=(0, 100000), ylims=None, smoothing=smoothing, interpolate=interpolate)

s_val = 31
smoothing = {'DAgger':111, 'Uniform':s_val, 'QBC':s_val}

plot_labels = ['Expert_Samples', 'Training Loss', 'SpaceInvaders-v0']
plot.plotData(data, plot_labels, data_axis=6, xlims=(0, 30000), ylims=None, smoothing=smoothing, interpolate=interpolate)

plot_labels = ['Expert_Samples', 'Action Similarity', 'SpaceInvaders-v0']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, 30000), ylims=None, smoothing=smoothing, interpolate=interpolate)

del data['DAgger']
plot_labels = ['Expert_Samples', 'Sample Utility', 'SpaceInvaders-v0']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, xlim), ylims=(0,1), smoothing=smoothing, interpolate=interpolate)

### OLD TESTS ###

# DAgger_fp = os.path.join(prefix, '0727/SpaceInvadersNoFrameskip-v0-classic-multi-dag_baseline_50epi_20ep-1.npy')
# # Random_fp = os.path.join(prefix, '0724/SpaceInvadersNoFrameskip-v0-pool-random-multi-1ep_b10.npy')
# Random_fp = os.path.join(prefix, '0728/SpaceInvadersNoFrameskip-v0-pool-random-multi-1ep_b30.npy')
# JSD_fp = os.path.join(prefix, '0728/SpaceInvadersNoFrameskip-v0-pool-multi-1ep_b30.npy')