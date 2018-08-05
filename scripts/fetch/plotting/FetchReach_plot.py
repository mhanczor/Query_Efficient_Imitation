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

xlim = 1000

#### FINAL AGENTS #####
DAgger_Concrete = os.path.join(prefix, 'Suc_0805/FetchReach-v1-classic-concrete-multi-suc_1e4_50ep_trim.npy')
Uncertainty = os.path.join(prefix, 'Suc_0805/FetchReach-v1-pool-concrete-multi-suc_1e4_10ep.npy')
Uncertainty_DW = os.path.join(prefix, 'Suc_0805/FetchReach-v1-pool-concrete-multi-suc_1e4_10ep_dw1-0.npy')
Random_Concrete = os.path.join(prefix, 'Suc_0805/FetchReach-v1-pool-random-concrete-multi-suc_1e4_10ep.npy')


filepaths = {'DAgger':DAgger_Concrete, 
            'Uniform':Random_Concrete,
            'Uncertainty':Uncertainty, #'Conc_DW_Long':conc_dw_long}
            'Uncertainty-DW':Uncertainty_DW}
sval = 7
interpolate = ['DAgger', 'Uniform', 'Uncertainty', 'Uncertainty-DW']
smoothing = {'DAgger_Concrete':0, 
            'Uncertainty':sval,
            'Uncertainty-DW':sval,
            'Uniform':sval}
# filepaths = {'Concrete':concrete_fp}


data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr
    
plot_labels = ['Expert_Samples', 'Episode Reward', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, xlim), ylims=None, interpolate=interpolate, smoothing=smoothing)
plot_labels = ['Expert_Samples', 'Success Rate', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0, xlim), ylims=None, smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, 2000), ylims=None, smoothing=None)

plot_labels = ['Expert_Samples', 'Training Loss', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=(50, 2000), ylims=(-1, 50), smoothing=None)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, 2000), ylims=None, smoothing=None)




# Old file paths
# 
# Pre-concrete dropout baselines
# DAgger_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-classic-multi.npy')
# Random_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-pool-random-multi.npy')



# test_fp = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/0714/FetchReach-v1-pool-multi.npy'
# concrete_fp = os.path.join(prefix, '0714/FetchReach-v1-pool-concrete-multi/FetchReach-v1-pool-concrete-multi.npy')
# concrete_2 = os.path.join(prefix, '0715/FetchReach-v1-pool-concrete-multi-0/FetchReach-v1-pool-concrete-multi.npy')
# concrete_random = os.path.join(prefix, '0715/FetchReach-v1-pool-random-concrete-multi/FetchReach-v1-pool-random-concrete-multi.npy')


# DAgger_fp = os.path.join(prefix, 'Baselines/FetchReach-v1-classic-concrete-multi-baseline.npy')
# Random_fp = os.path.join(prefix, '0715/FetchReach-v1-pool-random-concrete-multi-long/FetchReach-v1-pool-random-concrete-multi-long.npy')

# rand_basic = os.path.join(prefix, '0719/FetchReach-v1-pool-random-multi-basic.npy')

# dw_05 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_0-5.npy')
# dw_10 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_1-0.npy')
# dw_5e4 = os.path.join(prefix, '0719/FetchReach-v1-pool-concrete-multi-dw_5e4.npy')

# Conc_fp = os.path.join(prefix, '0720/FetchReach-v1-pool-concrete-multi-1e4-0.npy')
# Conc_dw_fp = os.path.join(prefix, '0720/FetchReach-v1-pool-concrete-multi-dw_1e4-0.npy')

# rand_basic_1e5 = os.path.join(prefix, '0719/FetchReach-v1-pool-random-multi-basic_1e5.npy')

# conc_dw_batch = os.path.join(prefix, '0721/FetchReach-v1-pool-concrete-multi-dw_1e4_batch-change-1.npy')
# conc_batch = os.path.join(prefix, '0721/FetchReach-v1-pool-concrete-multi-1e4_batch-change-1.npy')



# DAgger_fp = os.path.join(prefix, '0720/FetchReach-v1-classic-multi-1e5_baseline.npy')
# 
# ## Tests
# # Concrete_fp = os.path.join(prefix, '0715/FetchReach-v1-pool-concrete-multi-long/FetchReach-v1-pool-concrete-multi-long.npy')
# 
# rand_long = os.path.join(prefix, '0720/FetchReach-v1-pool-random-multi-1e5_baseline.npy')
# 
# conc_long = os.path.join(prefix, '0723/FetchReach-v1-pool-concrete-multi-1e4_baseline_10ep.npy')
# conc_dw_long = os.path.join(prefix, '0723/FetchReach-v1-pool-concrete-multi-1e4_dw_baseline_10ep.npy')

# Random_Concrete = os.path.join(prefix, '0801/FetchReach-v1-pool-random-concrete-multi-1ep_baseline_1e4.npy')



# DAgger = os.path.join(prefix, '0801/FetchReach-v1-classic-multi-dag_1e4_50ep_baseline_801.npy')
# DAgger_Concrete = os.path.join(prefix, '0801/FetchReach-v1-classic-concrete-multi-dag_1e4_50ep_baseline_801.npy')
# Uncertainty = os.path.join(prefix, '0803/FetchReach-v1-pool-concrete-multi-10ep_5e4ls_1e4lr_fc.npy')
# Uncertainty_DW = os.path.join(prefix, '0803/orig/FetchReach-v1-pool-concrete-multi-1e4_dw1-0_10ep_0803.npy')
# # Random = os.path.join(prefix, '0801/FetchReach-v1-pool-random-multi-1ep_baseline_1e4.npy')
# Random_Concrete = os.path.join(prefix, '0803/FetchReach-v1-pool-random-concrete-multi-10ep_5e4ls_1e4lr_fc.npy')


