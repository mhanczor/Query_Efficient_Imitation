import numpy as np
import tensorflow as tf
from active_imitation.utils import denseNet, concreteNet
import os


DEFAULT_PARAMS = {
    # 'layers': [16, 16, 16], # Layers and hidden units in network
    'lr': 0.001, # Learning rate
    'max_a': 1., # max absolute value of actions
    # 'dropout_rate': 0.1, # Dropout rate during training and forward samples
    'filepath': '~/Research/experiments/tmp/'
}

class GymRobotAgent(object):
    
    def __init__ (self, env_dims, layers, max_a, lr, 
                    dropout_rate, concrete, filepath='tmp/', load=False):
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
                
        self.env_dims = env_dims
        self.layers = layers
        self.max_a = max_a
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.filepath = filepath
        
        self.total_samples = 1
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        self.concrete = concrete
        if self.concrete:
            self._build_concrete_network()
        else:
            self._build_network()
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        
        if load:
            self._load_model()
        else:
            self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
            
    def _build_network(self):
                    
        o_dim = self.env_dims['observation']
        g_dim = self.env_dims['goal']
        a_dim = self.env_dims['action']

        self.dropout = tf.Variable(self.dropout_rate, name='Dropout_Rate')
        self.apply_dropout = tf.placeholder(tf.bool)
        
        self.o = tf.placeholder(tf.float32, [None, o_dim])
        self.g = tf.placeholder(tf.float32, [None, g_dim])
        policy_input = tf.concat(axis=1, values=[self.o, self.g]) # Concatenate observations and goals as a single network input
        network = denseNet(policy_input, self.layers, self.dropout, self.apply_dropout, name='Model')
        self.policy = self.max_a * tf.layers.dense(inputs=network, units=a_dim, activation=tf.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        with tf.name_scope("Loss"):
            # Continuous action spaces, MSE loss
            self.expert_action = tf.placeholder(tf.float32, [None, a_dim], name='Expert_Action')
            self.reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
            # self.loss = tf.losses.mean_squared_error(self.expert_action, self.policy)
            self.loss = tf.reduce_sum((self.expert_action - self.policy)**2. + self.reg_losses, -1)
        with tf.name_scope("Opt"):
            # Adam optimzer with a fixed lrz
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)    
            
        assert len(tf.losses.get_regularization_losses()) == len(self.layers) + 1, print(len(tf.losses.get_regularization_losses()))
        
    def _build_concrete_network(self):
        
        o_dim = self.env_dims['observation']
        g_dim = self.env_dims['goal']
        a_dim = self.env_dims['action']
        
        self.dropout = tf.Variable(self.dropout_rate, name='Dropout_Rate')
        self.apply_dropout = tf.placeholder(tf.bool) # Just keeping this in for no good reason
        
        self.o = tf.placeholder(tf.float32, [None, o_dim])
        self.g = tf.placeholder(tf.float32, [None, g_dim])
        policy_input = tf.concat(axis=1, values=[self.o, self.g])
        
        from active_imitation.utils import ConcreteDropout
        N = policy_input.shape[0]
        
        l = 1e-4
        self.N = tf.placeholder(tf.float32, [])
        wd = l**2./self.N
        dd = 2./self.N
        network = concreteNet(policy_input, self.layers, wd, dd, name='Model')
        self.policy = self.max_a * ConcreteDropout(tf.layers.Dense(units=a_dim, activation=tf.tanh, name='Policy_Mean'),
                                    weight_regularizer=wd, dropout_regularizer=dd)(network, training=True)
        
        log_var = ConcreteDropout(tf.layers.Dense(units=a_dim, name='Policy_Log_Var'), weight_regularizer=wd, 
                                        dropout_regularizer=dd)(network, training=True)
        
        self.prediction = tf.concat([self.policy, log_var], -1, name='Main_Output')                            
        
        def heteroscedastic_loss(true, pred):
            mean = pred[:, :a_dim] # Just separating out the concatenation from above
            log_var = pred[:, a_dim:]
            precision = tf.exp(-log_var)
            self.reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
            return tf.reduce_sum(precision * (true - mean)**2. + log_var + self.reg_losses, -1)                        
        
        self.expert_action = tf.placeholder(tf.float32, [None, a_dim], name='Expert_Action')
        self.loss = heteroscedastic_loss(self.expert_action, self.prediction)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        assert len(tf.losses.get_regularization_losses()) == len(self.layers) + 2, print(len(tf.losses.get_regularization_losses()))
        
    def update(self, batch):
        """
        Update the parameters of the network using the aggregated dataset of
        labeled expert examples
        
        batch gets fed in as an array of input and expert actions
        """   
        assert self.total_samples > 0
        
        feed_dict = {self.o:batch['observation'], self.g:batch['goal'], 
                    self.expert_action:batch['action'], self.apply_dropout:True}
        if self.concrete: 
            feed_dict[self.N] = self.total_samples
        _,loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
        loss = np.mean(loss) # For concrete
        return loss
        
    def updateDropout(self):
        """
        Modify the dropout rate if necessary
        
        This will ideally be used for Concrete Dropout at some point
        """
        raise NotImplementedError
    
    def _samplePolicy(self, state, apply_dropout):
        """
        Make a forward pass through the policy network
        """
        assert self.total_samples > 0
        feed_dict = {self.o:state['observation'], self.g:state['goal'], 
                    self.apply_dropout:apply_dropout}
        if self.concrete: 
            feed_dict[self.N] = self.total_samples
            
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
        
        action = self._samplePolicy(state, apply_dropout=apply_dropout)
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
        
        return action_avg, per_action_var
        
        
    def concreteAction(self, state, batch=32): 
        # This could be used to get the predicted variance along with the predicted mean
           
        pass
        
        
    
    def save_model(self, expert_samples=-1):
        savefile = os.path.join(self.filepath, 'checkpoints/model-'+ str(expert_samples) + '_samples.ckpt')
        self.saver.save(self.sess, savefile)
        print('Saved Model as {}'.format(savefile))
    
    
    def _load_model(self):
        loadfile = os.path.join(self.filepath, 'model.ckpt')
        self.saver.restore(self.sess, loadfile)
        print('Model Loaded')



    