import numpy as np
import gym

def saveStateImage(env, states, filepath='./', state_no=0):
    """
    Save a single image of a state or a group of images
    Input: env - Gym Environment
           state - numpy array of a state states to save
           filepath - the directory where the images should be saved
           state_no - where to start the state number labeling
    """
    states = np.atleast_2d(states)
    env.reset()
    for row in range(states.shape[0]):
        state = states[row,:,None] #TODO this may make the states a column not a row(do we need 2D anyways)
        