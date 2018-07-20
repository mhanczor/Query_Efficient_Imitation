import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPush Expert Perfomance
expert_valid = -8.908
expert_success = 0.999


# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPush-v1/'
DAgger_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-concrete-multi-baseline.npy')
DAgger_400_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-concrete-multi-400eps-baseline.npy')
Old_DAgger_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-multi.npy')

Random_conc_fp = os.path.join(prefix, '0716/FetchPush-v1-pool-random-concrete-multi-2samples.npy')
Dag_e4_e4_fp = os.path.join(prefix, '0717/FetchPush-v1-classic-concrete-multi-1e4-1e4-baseline.npy')
Dag_e4_e5_fp = os.path.join(prefix, '0717/FetchPush-v1-classic-concrete-multi-1e4-1e5-baseline.npy')
# Random_fp = os.path.join(prefix, 'FetchPush-v1-pool-random-multi.npy')

filepaths = {'DAgger':DAgger_fp, 'DAgger_400':DAgger_400_fp, 'Old_DAgger':Old_DAgger_fp, 
            'Random Concrete':Random_conc_fp, 'DAgger e4e4':Dag_e4_e4_fp, 'DAgger e4e5':Dag_e4_e5_fp}
# smoothing = {'Random':11, 'DAgger':11}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr    

plot_labels = ['Expert_Samples', 'Validation Reward', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, ylims=None, interpolate=False, smoothing=None)

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, ylims=(0.0, 1.0), interpolate=True, smoothing=None)


"""
REMOVING SOME OF THE DATSETS HERE, MAKE SURE TO CHANGE THIS LATER!!!!
"""
filepaths = {'DAgger':DAgger_fp, 'DAgger_400':DAgger_400_fp, 
            'Random Concrete':Random_conc_fp, 'DAgger e4e4':Dag_e4_e4_fp, 'DAgger e4e5':Dag_e4_e5_fp}
# smoothing = {'Random':11, 'DAgger':11}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr    

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchPush-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=None, ylims=None, smoothing=None)
# 
s_val = 15
# Only plot Random and Concrete here? Why?
smoothing = {'Rand Conc':s_val}#'DAgger':s_val, 'Random':s_val, 'Concrete':s_val}
plot_labels = ['Expert_Samples', 'Training Loss', 'FetchPush-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=None, ylims=(0, 60), smoothing=None)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchPush-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=None, ylims=(0, 2), smoothing=None)