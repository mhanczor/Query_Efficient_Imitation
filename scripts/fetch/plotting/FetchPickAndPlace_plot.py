import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPickAndPlace Expert Perfomance
expert_valid = -12.073
expert_success = 0.906


# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPickAndPlace-v1/'

DAgger_fp = os.path.join(prefix, 'Baselines/FetchPickAndPlace-v1-classic-concrete-multi-baseline.npy')
Old_DAgger_fp = os.path.join(prefix, 'Baselines/FetchPickAndPlace-v1-classic-multi.npy')
Dag_e4e4_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e4-baseline.npy')
Dag_e4e5_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e5-baseline.npy')
Dag_e4e5_lu_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e5-long_update.npy')
Rand_conc_fp = os.path.join(prefix, '0716/FetchPickAndPlace-v1-pool-random-concrete-multi-2samples.npy')

Conc_fp = os.path.join(prefix, '0718/FetchPickAndPlace-v1-pool-concrete-multi-10samp.npy')
Rand_Conc = os.path.join(prefix, '0718/FetchPickAndPlace-v1-pool-random-concrete-multi-10samp.npy')

filepaths = {'Concrete DAgger':DAgger_fp, 'Old DAgger':Old_DAgger_fp, 'Rand Conc':Rand_conc_fp, 'Concrete Dropout':Conc_fp, 'RandConcNew':Rand_Conc}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr
    
s_val = 21
smoothing = {'Rand Conc':s_val}

plot_labels = ['Expert_Samples', 'Validation Reward', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, 5000), ylims=None, smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0, 5000), ylims=None, smoothing=smoothing)

"""
REMOVING SOME OF THE DATSETS HERE, MAKE SURE TO CHANGE THIS LATER!!!!
"""
filepaths = {'Concrete DAgger':DAgger_fp, 'Rand Conc':Rand_conc_fp, 'Concrete Dropout':Conc_fp, 'RandConcNew':Rand_Conc}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, 5000), ylims=None, smoothing=None)
# 
s_val = 15
# Only plot Random and Concrete here? Why?
smoothing = {'Rand Conc':s_val}#'DAgger':s_val, 'Random':s_val, 'Concrete':s_val}
plot_labels = ['Expert_Samples', 'Training Loss', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=(0, 5000), ylims=(0, 60), smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, 5000), ylims=(0, 2), smoothing=smoothing)