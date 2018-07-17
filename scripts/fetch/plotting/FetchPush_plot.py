import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPush Expert Perfomance
expert_valid = -8.908
expert_success = 0.999


# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPush-v1/Baselines'
DAgger_fp = os.path.join(prefix, 'FetchPush-v1-classic-concrete-multi-baseline.npy')
Random_fp = os.path.join(prefix, 'FetchPush-v1-classic-multi.npy')
# Random_fp = os.path.join(prefix, 'FetchPush-v1-pool-random-multi.npy')

filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp}
smoothing = {'Random':11, 'DAgger':11}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr    

plot_labels = ['Expert_Samples', 'Validation Reward', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, ylims=None, interpolate=False, smoothing=None)

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, ylims=(0.0, 1.0), interpolate=True, smoothing=None)


