import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchPickAndPlace Expert Perfomance
expert_valid = -11.277 #-12.073
expert_success = 0.944 #0.906

xlim = 1400

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchPickAndPlace-v1/'
DAgger = os.path.join(prefix, 'Suc_0805/FetchPickAndPlace-v1-classic-concrete-multi-suc_10ep_1e4_comb.npy')
Uncertainty = os.path.join(prefix, 'Suc_0805/FetchPickAndPlace-v1-pool-concrete-multi-suc_10ep_1e4_comb_trim.npy')
Uncertainty_dw = os.path.join(prefix, 'Suc_0805/FetchPickAndPlace-v1-pool-concrete-multi-suc_1e4_10ep_dw1-0_comb.npy')
Random_Concrete = os.path.join(prefix, 'Suc_0805/FetchPickAndPlace-v1-pool-random-concrete-multi-suc_10ep_1e4_comb.npy')

filepaths = {'DAgger':DAgger, 'Uniform':Random_Concrete,
            'Uncertainty':Uncertainty, 'Uncertainty-DW':Uncertainty_dw}

interpolate = ['DAgger', 'Uniform', 'Uncertainty', 'Uncertainty-DW']

sval = 13
smoothing = {'DAgger':3, 'Uniform':sval,
            'Uncertainty':sval, 'Uncertainty-DW':sval}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr
    

plot_labels = ['Expert_Samples', 'Episode Reward', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, xlim), ylims=None, interpolate=interpolate, smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Success Rate', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0, xlim), ylims=None, interpolate=interpolate, smoothing=smoothing)


plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, xlim), ylims=None, smoothing=None)
# 
# s_val = 15
# Only plot Random and Concrete here? Why?
plot_labels = ['Expert_Samples', 'Training Loss', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=(0, xlim), ylims=(0, 60), smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, xlim), ylims=(0, 2), smoothing=smoothing)



### Old Tests
# Old_DAgger_fp = os.path.join(prefix, 'Baselines/FetchPickAndPlace-v1-classic-multi.npy')
# DAgger_fp = os.path.join(prefix, 'Baselines/FetchPickAndPlace-v1-classic-concrete-multi-baseline.npy')

# Dag_e4e4_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e4-baseline.npy')
# Dag_e4e5_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e5-baseline.npy')
# Dag_e4e5_lu_fp = os.path.join(prefix, '0717/FetchPickAndPlace-v1-classic-concrete-multi-1e4-1e5-long_update.npy')

# Rand_conc_fp = os.path.join(prefix, '0716/FetchPickAndPlace-v1-pool-random-concrete-multi-2samples.npy')
# DAgger_fp = os.path.join(prefix, '0721/FetchPickAndPlace-v1-classic-multi-baseline_1e5-50ep.npy') #actually 1e4

# Random = os.path.join(prefix, '0721/FetchPickAndPlace-v1-pool-random-multi-nonc_baseline_1e4.npy')
# Random_2 = os.path.join(prefix, '0722/FetchPickAndPlace-v1-pool-random-multi-nonc_baseline_1ep-1e4.npy')

# DAgger = os.path.join(prefix, '0721/FetchPickAndPlace-v1-classic-multi-baseline_1e4-10ep.npy')
# DAgger = os.path.join(prefix, '0801/FetchPickAndPlace-v1-classic-concrete-multi-50ep_baseline_1e4.npy')
