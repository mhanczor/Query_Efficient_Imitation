import active_imitation.utils.plot as plot
import numpy as np
import os

#LunarLander Expert Perfomance
expert_valid = 224.02

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/LunarLander-v2/'
DAgger_fp = os.path.join(prefix, 'Baselines/LunarLander-v2-classic-multi.npy')
# Random_fp = os.path.join(prefix, 'Baselines/LunarLander-v2-pool-random-multi.npy')

qbc_05 = os.path.join(prefix, '0712/LunarLander-v2-pool-DO-0-05.npy')

Random_fp = os.path.join(prefix, '0718/LunarLander-v2-pool-random-multi-baseline.npy')
DW_only_fp = os.path.join(prefix, '0718/LunarLander-v2-pool-random-multi-dw_baseline.npy')
JSD_DW_fp = os.path.join(prefix, '0718/LunarLander-v2-pool-multi-jsd-dw_baseline.npy')
JSD_DW_obj_fp = os.path.join(prefix, '0718/LunarLander-v2-pool-concrete-multi-jsd-dw-obj_baseline.npy')
bump_fp = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-concrete-multi-dw_lr_bump/LunarLander-v2-pool-concrete-multi-dw_lr_bump.npy'

filepaths = {'Random':Random_fp, 'DW Only':DW_only_fp,
            'JSD-DW':JSD_DW_fp, 'JSD-DW Obj':JSD_DW_obj_fp,
            'Bump':bump_fp}


# filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp, 'QBC-JSB 0.00':jsd_0, 'QBC-JSB 0.00-1':jsd_1}

data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot_labels = ['Expert_Samples', 'Validation Reward', 'LunarLander-v2']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, 300), ylims=(-600, 300), interpolate=False)

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, 300), ylims=None, smoothing=None)

s_val = 15
# Only plot Random and Concrete here? Why?
smoothing = {'DAgger':s_val, 'Random':s_val, 'Concrete':s_val}
plot_labels = ['Expert_Samples', 'Training Loss', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=(0, 300), ylims=None, smoothing=None)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchPickAndPlace-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, 300), ylims=None, smoothing=None)

### OLD TESTS ###
# dropout_tests = '/home/hades/Research/Active_Imitation/active_imitation/tests/LunarLander-v2/0712/'
# qbc_3 = dropout_tests + 'LunarLander-v2-pool-DO-0-3.npy'
# qbc_2 = dropout_tests + 'LunarLander-v2-pool-DO-0-2.npy'
# qbc_1 = dropout_tests + 'LunarLander-v2-pool-DO-0-1.npy'
# qbc_02 = dropout_tests + 'LunarLander-v2-pool-DO-0-02.npy'
# qbc_005 = dropout_tests + 'LunarLander-v2-pool-DO-0-005.npy'
# test_qbc_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-DO-0-05.npy'



# jsd_0 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-00.npy'
# jsd_1 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-00-1.npy'

# test_JSD_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-NonRand-0-05.npy'
# test_JSD_rand_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-05.npy'