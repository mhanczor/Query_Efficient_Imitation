import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import sem
import os


class confidencePlot(object):
    
    def __init__(self, y_label='Y-Axis', x_label='X-Axis', title='Confidence Plot'):
        self.y_label = y_label
        self.x_label = x_label
        
        self.figure = plt.figure(figsize=(12,9))
        self.ax = plt.subplot(111)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        
    def addSample(data):
        """
        This produces a line plot with confidence intervals as specified
        Uses the standard error of the mean as confidence intervals
        Inputs:
            data - N+1xM array of columns where the first column is the x-axis, and 
                   every other column is a dataset to use for predicting confidence
        """
        pass
    
def interpolateData(data, data_axis=4, max_samples=None):
    """
    Every run may not have an expert sample at intervals of 1
    Linearly interpolate the data between expert samples such at every interval
    there is an expert sample over which confidence bounds can be created
    
    Data - Nx2 array, first column is the number of expert samples, 2nd is the plotted value
    Data is expected to have a value 0 in it's [0,0] position
    """
    
    assert data[0,1,0] == 0
    
    # Set the max number of samples
    if max_samples == None:
        max_samples = int(np.min(data[-1, 1, :])) #TODO Fix this Largest number of expert samples available
    
    # import ipdb; ipdb.set_trace()
    new_data = np.empty((max_samples+1, data.shape[1], data.shape[2]))
    
    for k in range(new_data.shape[2]):
        assert int(data[-1, 1, k]) >= max_samples # Don't want to extrapolate
        i = 0
        for j in range(new_data.shape[0]):
            new_data[j, 1, k] = j # Number of expert samples
            if data[i, 1, k] == j: # if the expert samples at the current data is == to the value we want
                new_data[j, data_axis, k] = data[i, data_axis, k]
                i += 1
            else:
                interp = data[i-1, data_axis, k] + (j - data[i-1, 1, k]) * ((data[i, data_axis, k] - data[i-1, data_axis, k])/(data[i,1, k] - data[i-1, 1, k]))
                new_data[j, data_axis, k] = interp
            
    return new_data


def formatConfidenceData(data, bound='std', data_axis=4):
    """
    data - episode_len X 4 X num_of_runs numpy array with columns:
            [episode, expert_samples, validation_reward, variable stat]
            where variable stat is either the classifcation accuracy or avg successes per episode
    """    
    
    x_axis = data[:,1, 0]  #TODO This assumes that all the x-axis is the same, which it should be
    mean = np.mean(data[:,data_axis, :], axis=1, keepdims=False)
    
    if bound == 'sem':
        confidence_bound = sem(data[:, data_axis,:], axis=1)
    elif bound == 'std':
        confidence_bound = np.std(data[:, data_axis, :], axis=1) * 1 #TODO change back to 2
    else:
        raise ValueError
    
    return x_axis, mean, confidence_bound
    
def plotData(data, labels=None, data_axis=4, expert=None, xlims=None, ylims=None, interpolate=False):
    """
    This produces a line plot with confidence intervals as specified
    Uses the standard error of the mean as confidence intervals
    Args:
        data[dict] - Dict of multi_dim array of N+1xM array of columns where the 
            second column is the x-axis, and every other column is a dataset 
        labels[list] - String list of [x-axis, y-axis, title]
        expert[scalar] - Value that the expert/demonstrator performed at (horiz line on plot)
    """
    assert len(data.keys()) < 8 # Only 8 colors available in this set, don't expect this to be a problem

    # Set up the plot
    plt.figure(figsize=(12,9))
    
    # Remove the plot frame lines
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set the axis ticks to only show up on the bottom and left of plot
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    # Limit the range of the plot to the data
    # plt.ylim(-60, 0)
    if xlims != None:
        plt.xlim(xlims[0], xlims[1])
    if ylims != None:
        plt.ylim(ylims[0], ylims[1])
        # plt.yticks(range(ylims[0], ylims[1], (ylims[1] - ylims[0])/10.), fontsize=14)
    
    # Increase axis tick marks
    # TODO set these tick marks to reflect the actual data, or let this be a keyword arg
    # plt.xticks(range(0, 100, 10), fontsize=14)
    # plt.yticks(range(-60, 0, 10), fontsize=14)
    
    # Axis and title labels with increased size
    if labels == None:
        labels = ['X-Axis', 'Y-Axis', 'Title']
    plt.xlabel(labels[0], fontsize=16)
    plt.ylabel(labels[1], fontsize=16)
    plt.title(labels[2], fontsize=22)
    
    # Pastel2 and Dark2 should be the colormaps used for confidence and mean resp.
    cmap = ['Dark2', 'Pastel2'] 
    
    # Iterate over the recorded datasets
    i = 0
    for name, values in data.items():
        if interpolate:
            values = interpolateData(values, data_axis=data_axis)
        x_axis, means, bands = formatConfidenceData(values, bound='std', data_axis=data_axis)
        plt.plot(x_axis, means, lw=2, label=name, color=cm.Dark2(i)) # color = ???
        plt.fill_between(x_axis, means - bands, means + bands, color=cm.Pastel2(i)) # color=???
        i += 0.126
    plt.legend(loc=4, prop={'size':20}) # There are up to 10 positions, 0 is best, 1 upper right, 4, lower right
    plt.show()
    
    
    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
    # plt.savefig(filename, bbox_inches="tight") 
        
"""
Want to be able to plot the data by passing in an arbitrary number of runs
Each line should be associated with a label that gets added to the legend
The data that gets passed in should be the raw data, and the confidence intervals
are calculated in the function

Don't need to make this too general, it's going to be used to plot 
"""
        
if __name__ == "__main__":
    
    # import ipdb; ipdb.set_trace()
    prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/Baselines'
    fp = 'FetchReach-v1-classic-multi-DAgger.npy'
    fp = os.path.join(prefix, fp)    
    dag_dat = np.load(fp)
    data = {'DAgger':dag_dat}
    labels = ['Expert Samples', 'Episode Return', 'FetchReach-v1']
    
    plotData(data, labels)
    
    
    