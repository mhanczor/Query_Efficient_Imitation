import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import sem
import os


import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    y = y[:x.shape[0]]
    return y

def interpolateData(data, data_axis=4, max_samples=None):
    """
    Every run may not have an expert sample at intervals of 1
    Linearly interpolate the data between expert samples such at every interval
    there is an expert sample over which confidence bounds can be created
    
    Data - Nx2 array, first column is the number of expert samples, 2nd is the plotted value
    Data is expected to have a value 0 in it's [0,0] position
    """
    
    assert data[0,1,0] == 0
    # import ipdb; ipdb.set_trace()
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
    
def plotData(data, labels=None, data_axis=4, expert=None, xlims=None, ylims=None, interpolate=None, smoothing=None):
    """
    This produces a line plot with confidence intervals as specified
    Uses the standard error of the mean as confidence intervals
    Args:
        data[dict] - Dict of multi_dim array of N+1xM array of columns where the 
            second column is the x-axis, and every other column is a dataset 
        labels[list] - String list of [x-axis, y-axis, title]
        expert[scalar] - Value that the expert/demonstrator performed at (horiz line on plot)
    """
    assert len(data.keys()) <= 8 # Only 8 colors available in this set, don't expect this to be a problem

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
    if xlims != None:
        plt.xlim(xlims[0], xlims[1])
    if ylims != None:
        plt.ylim(ylims[0], ylims[1])
        # plt.yticks(range(ylims[0], ylims[1], (ylims[1] - ylims[0])/10.), fontsize=14)
    
    # Increase axis tick marks
    # plt.xticks(range(0, 100, 10), fontsize=14)
    # plt.yticks(range(-60, 0, 10), fontsize=14)
    
    ax.tick_params(labelsize=16)
    
    # Axis and title labels with increased size
    if labels == None:
        labels = ['X-Axis', 'Y-Axis', 'Title']
    plt.xlabel(labels[0], fontsize=24)
    plt.ylabel(labels[1], fontsize=24)
    # plt.title(labels[2], fontsize=22) #### TODO WHEN MAKING SLIDE DECK
    
    # Show a horizontal line with the expert's performance if available
    if expert is not None:
        plt.hlines(expert, xmin=0, xmax=20000, colors='#737373', 
                label='Expert', linewidths=3, linestyles='dashdot') #linestyles : [‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’],
    
    # Pastel2 and Dark2 should be the colormaps used for confidence and mean resp.
    cmap = ['Dark2', 'Pastel2'] 
    
    # Iterate over the recorded datasets
    i = 0.0
    for name, values in data.items():
        if interpolate is not None and name in interpolate:
            values = interpolateData(values, data_axis=data_axis)
        if smoothing is not None and name in smoothing:
            for j in range(values.shape[2]):
                values[:,data_axis,j] = smooth(values[:,data_axis,j], window_len=smoothing[name])
        x_axis, means, bands = formatConfidenceData(values, bound='std', data_axis=data_axis)
        plt.plot(x_axis, means, lw=2, label=name, color=cm.Dark2(i)) 
        plt.fill_between(x_axis, means - bands, means + bands, color=cm.Pastel2(i))
        i += 0.126 # Change color
        
    plt.legend(loc=4, prop={'size':24}) # There are up to 10 positions, 0 is best, 1 upper right, 2 top left, 3, 4 lower right
    plt.show()

        
if __name__ == "__main__":
    
    # import ipdb; ipdb.set_trace()
    prefix = '/home/hades/Research/Active_Imitation/active_imitation/tests/FetchReach-v1/Baselines'
    fp = 'FetchReach-v1-classic-multi-DAgger.npy'
    fp = os.path.join(prefix, fp)    
    dag_dat = np.load(fp)
    data = {'DAgger':dag_dat}
    labels = ['Expert Samples', 'Episode Return', 'FetchReach-v1']
    
    plotData(data, labels)
    
    
    