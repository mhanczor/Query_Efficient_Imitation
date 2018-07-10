import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from active_imitation import gym_dagger


class confidencePlot(object):
    
    def __init__(self, y_label='Y-Axis', x_label='X-Axis', title='Confidence Plot'):
        self.y_label = y_label
        self.x_label = x_label
        
        self.figure = plt.figure(figsize=(12,9))
        self.ax = plt.subplot(111)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        
    def addSample(data, ):
        """
        This produces a line plot with confidence intervals as specified
        Uses the standard error of the mean as confidence intervals
        Inputs:
            data - N+1xM array of columns where the first column is the x-axis, and 
                   every other column is a dataset to use for predicting confidence
        """
    
def interpolateData(data, max_samples=None):
    """
    Every run may not have an expert sample at intervals of 1
    Linearly interpolate the data between expert samples such at every interval
    there is an expert sample over which confidence bounds can be created
    
    Data - Nx2 array, first column is the number of expert samples, 2nd is the plotted value
    Data is expected to have a value 0 in it's [0,0] position
    """
    
    assert data[0,0] == 0
    
    # Set the max number of samples
    if max_samples == None:
        max_samples = data[-1, 0] # Largest number of expert samples available
    assert data[-1,0] <= max_samples # Don't want to extrapolate
    
    new_data = np.empty((max_samples+1, 2))
    
    i = 0
    for j in range(new_data.shape[0]):
        new_data[j, 0] = j # Number of expert samples
        if data[i, 0] == j: # if the expert samples at the current data is == to the value we want
            new_data[j, 1] = data[i, 1]
            i += 1
        elif:
            interp = data[i-1, 1] + (j - data[i-1, 0]) * ((data[i, 1] - data[i-1, 1])/(data[i,0] - data[i-1, 0]))
            new_data[j, 1] = interp
            
    return new_data


def formatConfidenceData(data, bound='sem'):
    """
    data - first column is the x-axis, 
    """    
    
    x_axis = data[:,0]    
    mean = np.mean(data[:, 1:], axis=1)
    
    if bound == 'sem':
        confidence_bound = sem(data[:, 1:], axis=1)
    elif bound == 'std':
        confidence_bound = np.std(data[:, 1:], axis=1) * 2
    else:
        raise ValueError
    
    return x_axis, mean, confidence_bound
    
def plotData(data, labels=None):
    """
    This produces a line plot with confidence intervals as specified
    Uses the standard error of the mean as confidence intervals
    Inputs:
        data - Dictionary of N+1xM array of columns where the first column is the x-axis, and 
               every other column is a dataset to use for predicting confidence
               Keys of the dictionary are the labels
    """
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
    plt.ylim(63, 85)
    
    # Increase axis tick marks
    # TODO set these tick marks to reflect the actual data, or let this be a keyword arg
    plt.xticks(range(0, 300, 10), fontsize=14)
    plt.yticks(range(-200, 250, 10), fontsize=14)
    
    # Axis and title labels with increased size
    if labels == None:
        labels = ['X-Axis', 'Y-Axis', 'Title']
    plt.xlabel(labels[0], fontsize=16)
    plt.ylabel(labels[1], fontsize=16)
    plt.title(labels[2], fontszie=22)
    
    # Pastel2 and Dark2 should be the colormaps used for confidence and mean resp.
    cmap = ['Dark2', 'Pastel2'] 
    
    # Iterate over the recorded datasets
    for name, values in data.iteritems():
        x_axis, means, bands = formatConfidenceData(values, bound='sem')
        plt.plot(x_axis, means, lw=2, label=name) # color = ???
        plt.fill_between(x_axis, means - bands, means + bands) # color=???
    plt.show()
    
    
    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
    # plt.savefig(filename, bbox_inches="tight") 
  
# Always include your data source(s) and copyright notice! And for your  
# data sources, tell your viewers exactly where the data came from,  
# preferably with a direct link to the data. Just telling your viewers  
# that you used data from the "U.S. Census Bureau" is completely useless:  
# the U.S. Census Bureau provides all kinds of data, so how are your  
# viewers supposed to know which data set you used?  
# plt.xlabel("\nData source: www.ChessGames.com | "  
#            "Author: Randy Olson (randalolson.com / @randal_olson)", fontsize=10)  

    
        
"""
Want to be able to plot the data by passing in an arbitrary number of runs
Each line should be associated with a label that gets added to the legend
The data that gets passed in should be the raw data, and the confidence intervals
are calculated in the function

Don't need to make this too general, it's going to be used to plot 
"""
        
if __name__ == "__main__":
    
    import ipdb; ipbd.set_trace()
    for i in range(10):
        rewards, stats = gym_dagger.sampleRun()
    
    
    
    