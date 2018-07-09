import numpy as np
import tensorflow as tf
from active_imitation.utils import denseNet


DEFAULT_PARAMS = {
    'layers': [16, 16, 16], # Layers and hidden units in network
    'lr': 0.001, # Learning rate
    'max_a': 1., # max absolute value of actions
    'batch_size': 32, # Batches to use for updating network
    'dropout_rate': 0.1, # Dropout rate during training and forward samples
    'filepath': '~/Research/experiments/tmp/'
}

class GymRobotAgent(object):
    
    def __init__ (self, sess, env_dims, layers, max_a, batch_size, 
                    lr=0.1, dropout_rate=0.1, filepath='tmp/'):
        """
        Agent that learns via imitation to perform an OpenAI Robotic Task
            
        Args:
            sess(tf session): current tensorflow session
            env_dims (dict of ints): dimensions for the observatioins, the goal, 
                and actions
            layers (list of ints): number and size of hidden layers
            max_a (float): maximum action magnitude, clipped between [-max_a, max_a]
            lr (float): learning rate for the network
            dropout_rate[float]: Probability of dropout for any node, in range [0,1],
                                a value of 0 would lead to no dropout
            filepath[str]: policy and data save location 
        """
                
        o_dim = env_dims['observation']
        g_dim = env_dims['goal']
        a_dim = env_dims['action']
        
        self.sess = sess
        
        self.dropout = tf.Variable(dropout_rate, name='Dropout_Rate')
        self.apply_dropout = tf.placeholder(tf.bool)
        
        self.o = tf.placeholder(tf.float32, [None, o_dim])
        self.g = tf.placeholder(tf.float32, [None, g_dim])
        policy_input = tf.concat(axis=1, values=[self.o, self.g]) # Concatenate observations and goals as a single network input
        network = denseNet(policy_input, layers, self.dropout, self.apply_dropout, name='Model')
        self.policy = max_a * tf.layers.dense(inputs=network, units=a_dim, activation=tf.tanh)
        
        with tf.name_scope("Loss"):
            # Continuous action spaces, MSE loss
            self.expert_action = tf.placeholder(tf.float32, [None, a_dim], name='Expert_Action')
            self.loss = tf.losses.mean_squared_error(self.expert_action, self.policy)
        with tf.name_scope("Opt"):
            # Adam optimzer with a fixed lr
            self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(filepath+'events/', self.sess.graph)
        
    def update(self, batch):
        """
        Update the parameters of the network using the aggregated dataset of
        labeled expert examples
        
        batch gets fed in as an array of input and expert actions
        """   
        feed_dict = {self.o:batch['observation'], self.g:batch['goal'], 
                    self.expert_action:batch['action'], self.apply_dropout:True}
                    
        _,loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
        return loss
        
    def updateDropout(self):
        """
        Modify the dropout rate if necessary
        
        This will ideally be used for Concrete Dropout at some point
        """
        raise NotImplementedError
    
    def samplePolicy(self, state, apply_dropout):
        """
        Make a forward pass through the policy network
        """
        feed_dict = {self.o:state['observation'], self.g:state['goal'], 
                    self.apply_dropout:apply_dropout}
        return self.sess.run(self.policy, feed_dict=feed_dict)
    
    def sampleAction(self, state, batch=1, apply_dropout=False):
        """
        Sample an action from the policy based on the current state
        If using a batch, sample from multiple copies of the same state
        """
        # import ipdb; ipdb.set_trace()
        
        o = np.atleast_2d(state['observation'])
        g = np.atleast_2d(state['desired_goal'])
        o = np.repeat(o, batch, axis=0)
        g = np.repeat(g, batch, axis=0)
        state = {'observation':o, 'goal':g}
        
        action = self.samplePolicy(state, apply_dropout=apply_dropout)
        #TODO may want to flatten batch size 1 here instead of outside?
        return action
        
    def uncertainAction(self, state, batch=32):
        """
        Sample multiple actions from the same state through multiple stochastic
        forward passes.  Take the mean action and return the total variance
        over the action space.
        """
        actions = self.sampleAction(state, batch, apply_dropout=True)
        
        action_avg = np.mean(actions, axis=0, keepdims=True)
        per_action_var = np.var(actions, axis=0)
        
        # Assume independence between actions in the action space, sum variances
        action_var = np.sum(per_action_var)
        return action_avg, action_var



    