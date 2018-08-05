import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPush Expert Perfomance
expert_valid = -8.834
expert_success = 0.999


xlim = 10000

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPush-v1/'
DAgger_fp = os.path.join(prefix, '0722/FetchPush-v1-classic-multi-_baseline_last.npy')
Random_conc_fp = os.path.join(prefix, '0716/FetchPush-v1-pool-random-concrete-multi-2samples.npy')
Concrete = os.path.join(prefix, '0725/FetchPush-v1-pool-concrete-multi-1ep_baseline.npy')
Concrete_dw = os.path.join(prefix, '0729/FetchPush-v1-pool-concrete-multi-dw_baseline_1ep.npy')

Rand_latest = os.path.join(prefix, '0731/FetchPush-v1-pool-random-multi-1ep_baseline_1e5.npy')

filepaths = {'DAgger':DAgger_fp, 
            'Uniform':Random_conc_fp, 
            'Uncertianty':Concrete, 
            'Uncertianty_DW':Concrete_dw}
# smoothing = {'Random':11, 'DAgger':11}

sval = 11
smoothing = {'DAgger':0, 'Random Concrete':sval, 
            'Random':sval, 'Uncertianty':sval, 'Uncertianty_DW':sval}
            
interpolate = ['DAgger', 'Uniform', 'Uncertianty', 'Uncertianty_DW']

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr    

plot_labels = ['Expert_Samples', 'Episode Reward', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0,xlim), ylims=None, interpolate=interpolate, smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPush-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0,xlim), ylims=(0.0, 1.0), interpolate=interpolate, smoothing=smoothing)


"""
REMOVING SOME OF THE DATSETS HERE, MAKE SURE TO CHANGE THIS LATER!!!!
"""
# filepaths = {'DAgger':DAgger_fp, 'DAgger_400':DAgger_400_fp, 
#             'Random Concrete':Random_conc_fp, 'DAgger e4e4':Dag_e4_e4_fp, 'DAgger e4e5':Dag_e4_e5_fp}
# # smoothing = {'Random':11, 'DAgger':11}

# data = {}
# for name, filepath in filepaths.items():
#     arr = np.load(filepath)
#     data[name] = arr    

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



####
# Dag_e4_e4_fp = os.path.join(prefix, '0717/FetchPush-v1-classic-concrete-multi-1e4-1e4-baseline.npy')
# Dag_e4_e5_fp = os.path.join(prefix, '0717/FetchPush-v1-classic-concrete-multi-1e4-1e5-baseline.npy')
# DAgger_400_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-concrete-multi-400eps-baseline.npy')
# Old_DAgger_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-multi.npy')

# Hloss = os.path.join(prefix, '0722/FetchPush-v1-pool-multi-hloss_1e4-0.npy')
# Hloss_dw = os.path.join(prefix, '0722/FetchPush-v1-pool-multi-hloss_dw_1e4-0.npy')

# DAgger_fp = os.path.join(prefix, 'Baselines/FetchPush-v1-classic-concrete-multi-baseline.npy')

Random_fp = os.path.join(prefix, '0725/FetchPush-v1-pool-random-multi-1ep_baseline.npy')
