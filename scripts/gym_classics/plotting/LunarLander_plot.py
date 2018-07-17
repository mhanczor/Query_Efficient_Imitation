import active_imitation.utils.plot as plot
import numpy as np
import os

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/LunarLander-v2/Baselines'
DAgger_fp = os.path.join(prefix, 'LunarLander-v2-classic-multi.npy')
Random_fp = os.path.join(prefix, 'LunarLander-v2-pool-random-multi.npy')

test_qbc_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-DO-0-05.npy'

# test_JSD_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-NonRand-0-05.npy'
# test_JSD_rand_05 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-05.npy'
dropout_tests = '/home/hades/Research/Active_Imitation/active_imitation/tests/LunarLander-v2/0712/'
qbc_3 = dropout_tests + 'LunarLander-v2-pool-DO-0-3.npy'
qbc_2 = dropout_tests + 'LunarLander-v2-pool-DO-0-2.npy'
qbc_1 = dropout_tests + 'LunarLander-v2-pool-DO-0-1.npy'
qbc_05 = dropout_tests + 'LunarLander-v2-pool-DO-0-05.npy'
qbc_02 = dropout_tests + 'LunarLander-v2-pool-DO-0-02.npy'
qbc_005 = dropout_tests + 'LunarLander-v2-pool-DO-0-005.npy'

jsd_0 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-00.npy'
jsd_1 = '/home/hades/Research/Active_Imitation/scripts/gym_classics/tmp/LunarLander-v2/LunarLander-v2-pool-JSD-Rand-0-00-1.npy'

# 
filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp, 
            'QBC-KL 0.3':qbc_3, 'QBC-KL 0.2':qbc_2, 
            'QBC-KL 0.1':qbc_1, 'QBC-KL 0.05':qbc_05,
            'QBC-KL 0.02':qbc_02, 'QBC-KL 0.005':qbc_005}


# filepaths = {'DAgger':DAgger_fp, 'Random':Random_fp, 'QBC-JSB 0.00':jsd_0, 'QBC-JSB 0.00-1':jsd_1}

plot_labels = ['Expert_Samples', 'Episodic Return', 'LunarLander-v2']
data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr

plot.plotData(data, plot_labels, data_axis=3, xlims=(0, 300), ylims=(-800, 300), interpolate=True)