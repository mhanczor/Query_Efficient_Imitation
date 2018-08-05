import active_imitation.utils.plot as plot
import numpy as np
import os

#FetchReach Expert Perfomance
expert_valid = -1.709
expert_success = 1.0

# Stats are [0:Episode, 1:Total Expert Samples, 2:Expert Samples/Episode, 
#           3:Validation Reward, 4:Average Successes, 5:Uncertainty of Selected Sample, 
#           6:Final Training Loss]

random = False

# Classic DAgger
prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/'

#### FINAL AGENTS #####
DAgger_fp = os.path.join(prefix, '0720/FetchReach-v1-classic-multi-1e5_baseline.npy')

## Tests
# Concrete
conc_b_1 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-concrete-multi-b1_btests_2ep.npy')
conc_b_2 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-concrete-multi-b2_btests_4ep.npy')
conc_b_5 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-concrete-multi-b5_btests_10ep.npy')
conc_b_10 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-concrete-multi-b10_btests_20ep.npy')

# Random
rand_b_1 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-random-multi-b1_btests_2_ep.npy')
rand_b_2 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-random-multi-b2_btests_4_ep.npy')
rand_b_5 = os.path.join(prefix, 'budget_tests/FetchReach-v1-pool-random-multi-b5_btests.npy')
rand_b_10 = os.path.join(prefix, 'budget_tests_2/FetchReach-v1-pool-random-multi-b10_btests_20_ep.npy')


sval = 0
if random:
    filepaths = {'DAgger':DAgger_fp, 'Random B=1':rand_b_1, 'Random B=2':rand_b_2, 
                'Random B=5':rand_b_5, 'Random B=10':rand_b_10}
    smoothing = {'DAgger':0, 'Random B=1':sval, 'Random B=2':sval, 'Random B=5':sval, 
                'Random B=10':sval}
else:
    # filepaths = {'DAgger':DAgger_fp, 'Uncertainty B=1':conc_b_1, 'Uncertainty B=2':conc_b_2, 
                # 'Uncertainty B=5':conc_b_5, 'Uncertainty B=10':conc_b_10}
    filepaths = {'DAgger':DAgger_fp, 'Uncertainty B=1':conc_b_1, 'Uncertainty B=2':conc_b_2, 
                'Uncertainty B=5':conc_b_5}
    smoothing = {'DAgger':0, 'Uncertainty B=1':sval, 'Uncertainty B=2':sval, 
                'Uncertainty B=5':sval, 'Uncertainty B=10':sval}
            

# interpolate = ['DAgger']

# filepaths = {'Concrete':concrete_fp}


data = {}
for name, filepath in filepaths.items():
    arr = np.load(filepath)
    data[name] = arr
plot_labels = ['Expert_Samples', 'Episode Reward', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_valid, data_axis=3, xlims=(0, 1000), ylims=None, smoothing=smoothing)
plot_labels = ['Expert_Samples', 'Success Rate', 'FetchReach-v1']
plot.plotData(data, plot_labels, expert=expert_success, data_axis=4, xlims=(0, 1000), ylims=None, smoothing=smoothing)

plot_labels = ['Expert_Samples', 'Sample Uncertainty', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=5, xlims=(0, 2000), ylims=None, smoothing=None)

plot_labels = ['Expert_Samples', 'Training Loss', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=6, xlims=(50, 2000), ylims=(-1, 50), smoothing=None)

plot_labels = ['Expert_Samples', 'Action Similarity', 'FetchReach-v1']
plot.plotData(data, plot_labels, data_axis=7, xlims=(0, 2000), ylims=None, smoothing=None)





# rand_b_1
# rand_b_2 = os.path.join(prefix, 'budget_tests/FetchReach-v1-pool-random-multi-b2_btests.npy')
# rand_b_10 = os.path.join(prefix, 'budget_tests/FetchReach-v1-pool-random-multi-b10_btests.npy')

