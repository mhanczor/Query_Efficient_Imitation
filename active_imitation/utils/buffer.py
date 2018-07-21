import numpy as np

"""
Need to create a buffer that we can append data to as we collect expert samples

We consider a buffer for 3 different environment types:
    Classic Control [CartPole-v1]
        State - observations [float32]
        Action - discrete[int8]
        Q-value -  
    Box2D [LunarLander-v2]
        State - observations [float32]
    Atari [SpaceInvaders]
        State - observations[uint8]
        Action - discrete[int8]
    Robotics [Fetch, HandManipulation]
        State - observations[float32], goal[float32]
        Action - continuous[float32]
        Q-value - 
May want to allow for Q-values to be stored as well, save time later?


Buffer could have a robotics flag that gets used to extract the robotics stuff 

Buffer should have the following abilities:
    __init__
    store data(state, action, q=None)
    sample data(batch)
    

"""

class AggBuffer(object):
    
    def __init__(self, spaces, continuous=False, max_samples=None):
        """
        Args:  
            spaces[dict] - Contains the items to be aggregated and their shape (scalar)
            continuous[bool] - if the action space is continuous, otherwise assume int8
            max_sample[int] - largest size the buffer can take 
        """
        obs_space = (0,) + spaces['observation']
        if len(obs_space) == 4:
            self.buffer = {'observation':np.empty(obs_space, dtype=np.uint8),
                        'action':np.empty((0, spaces['action']))}
        else:
            self.buffer = {'observation':np.empty(obs_space),
                        'action':np.empty((0, spaces['action']))}
        self.store_q = False
        
        # If the state space has multiple inputs, the environment is a gym robot env
        if 'goal' in spaces:
            self.robot = True
            self.buffer['goal'] = np.empty((0, spaces['goal']))           
        else:
            self.robot = False
        
        if 'q_value' in spaces:
            self.store_q = True
            self.buffer['q_value'] = np.empty((0, spaces['q_value'])) 
        
    def store(self, state, action, q_val=None):
        """
        Aggregate expert samples to the dataset
        """
        if self.robot:
            self.buffer['goal'] = np.append(self.buffer['goal'], state['desired_goal'][None,:], axis=0)
            self.buffer['observation'] = np.append(self.buffer['observation'], state['observation'][None,:], axis=0)
        else:
            if state.ndim == 4:
                state = state[0, :, :, :]
            self.buffer['observation'] = np.append(self.buffer['observation'], state[None,:], axis=0)
            
        action = np.atleast_1d(action)
        self.buffer['action'] = np.append(self.buffer['action'], action[None,:], axis=0)
        
        if self.store_q:
            self.buffer['q_value'] = np.append(self.buffer['q_value'][None,:], q_val, axis=0)
    
    def sample(self, indices):
        """
        Select values to read from the dataset
        """
        datatypes = ['observation', 'action']
        if self.robot:
            datatypes.append('goal')
        if self.store_q:
            datatypes.append('q_value')
        
        sample_batch = {}
        for val in datatypes:
            sample_batch[val] = self.buffer[val][indices, :]
        
        return sample_batch
    
    @property
    def size(self):
        # Number of samples in the buffer
        return self.buffer['observation'].shape[0]
        
        
        
        