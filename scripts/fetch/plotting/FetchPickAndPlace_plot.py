import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPickAndPlace Expert Perfomance
expert_valid = -12.073
expert_success = 0.906


# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPickAndPlace-v1/Baselines'
DAgger_fp = os.path.join(prefix, 'FetchPickAndPlace-v1-classic-multi.npy')

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPickAndPlace-v1']

filepaths = {'DAgger':DAgger_fp}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot.plotData(data, plot_labels, ylims=(0.0, 1.0))