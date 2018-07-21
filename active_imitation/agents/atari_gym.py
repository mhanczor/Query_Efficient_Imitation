import numpy as np
import tensorflow as tf
from scipy import stats
import os

from active_imitation.utils import denseNet, concreteNet

DEFAULT_PARAMS = {
    # 'layers': [16, 16, 16], # Layers and hidden units in network
    'lr': 0.001, # Learning rate
    # 'dropout_rate': 0.1, # Dropout rate during training and forward samples
    'filepath': '~/Research/experiments/tmp/'
}

class AtariGymAgent(object):
    
    def __init__(self, env_dims, layers, lr, dropout_rate, 
                concrete, filepath='tmp/', **kwargs):
        """
        Learner agent for OpenAI Gym's classic environments like CartPole and LunarLander
        
        Args:
            sess(tf session): current tensorflow session
            env_dims (dict): Contains the size of the observation and action spaces
            lr (float): learning rate for the network
            dropout_rate[float]: Probability of dropout for any node, in range [0,1],
                                a value of 0 would lead to no dropout
            filepath[str]: policy and data save location        
        """
        
        self.env_dims = env_dims
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.filepath = filepath
        
        self.total_samples = 1
        
        self.sess = tf.Session()
        # if self.sess is None:
        #     self.sess =  tf.InteractiveSession()
        
        self.hetero_loss = concrete
        self._build_network()
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        
        self.writer = tf.summary.FileWriter(filepath+'events/', self.sess.graph)
        
    def _build_network(self):
        
        wr = 1e-4
        input_size = (None,) + self.env_dims['observation']
        output_size = self.env_dims['action_space']
        
        with tf.name_scope("Inputs"):
            self.state = tf.placeholder(tf.float32, input_size, name='State')
            self.expert_action = tf.placeholder(tf.int32, [None, ], name='Expert_Action')
            self.apply_dropout = tf.placeholder(tf.bool)
        
        scaled_obs = tf.cast(self.state, tf.float32) / 255.
        conv_1 = tf.layers.conv2d(inputs=scaled_obs, filters=32, kernel_size=8, strides=4,
                            kernel_initializer=None, activation=tf.nn.relu)
        conv_2 = tf.layers.conv2d(inputs=conv_1, filters=64, kernel_size=4, strides=2,
                            kernel_initializer=None, activation=tf.nn.relu)
        conv_3 = tf.layers.conv2d(inputs=conv_2, filters=64, kernel_size=3, strides=1,
                            kernel_initializer=None, activation=tf.nn.relu)
        nh = np.prod([v.value for v in conv_3.get_shape()[1:]])
        flat_c3 = tf.reshape(conv_3, [-1, nh])

        fc_layers = denseNet(flat_c3, self.layers, self.dropout_rate, self.apply_dropout, reg_weight=wr, name='FC_Layers')
        logits = tf.layers.dense(inputs=fc_layers, units=output_size, kernel_initializer=tf.random_normal_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(wr))
        self.policy = tf.nn.softmax(logits, name='Policy_output')
        self.log_var = tf.layers.dense(inputs=fc_layers, units=output_size, kernel_initializer=tf.random_normal_initializer(1.0), kernel_regularizer=tf.contrib.layers.l2_regularizer(wr))

        with tf.name_scope("Loss"):
            self.reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
            labels = tf.one_hot(self.expert_action, output_size, axis=-1)
            self.ce_loss = tf.losses.softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.NONE) 
            
            if self.hetero_loss:
                self.ce_loss = tf.reshape(self.ce_loss, [tf.shape(self.ce_loss)[0],1])
                precision = tf.exp(-self.log_var)
                self.loss = tf.reduce_mean(tf.reduce_sum(precision*self.ce_loss + self.log_var + self.reg_losses, -1),-1)
                # self.loss = self.ce_loss + self.reg_losses
            else:
                self.ce_loss = tf.reduce_sum(self.ce_loss, -1)#tf.reduce_mean(self.ce_loss, -1)
                self.loss = self.ce_loss + self.reg_losses
            # self.loss = self.ce_loss + self.reg_losses
            # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.name_scope("Opt"):
            # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            train_opt = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = train_opt.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grad_val = tf.Print(grad, [tf.norm(grad), tf.norm(var), tf.norm(tf.clip_by_norm(grad, 10))])
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, 10), var)
            self.opt = train_opt.apply_gradients(grads_and_vars)
        assert len(tf.losses.get_regularization_losses()) == len(self.layers) + 2, print(len(tf.losses.get_regularization_losses()))
    
    def update(self, batch):
        feed_dict = {self.state:batch['observation'], self.expert_action:batch['action'].flatten(), 
                    self.apply_dropout:True}
        
        # import ipdb; ipdb.set_trace()        
        _, loss, reg_loss, ce_loss, log_var = self.sess.run([self.opt, self.loss, self.reg_losses, self.ce_loss, self.log_var], feed_dict=feed_dict)
        # print("Cross Entropy Loss: {} \nPrecision: {} \nReg. Loss: {} \nTotal Loss: {}".format(ce_loss, precision, reg_loss, loss))
        return loss
    
    def _samplePolicy(self, state, apply_dropout=False):
        state = np.atleast_2d(state)
        feed_dict = {self.state : state, self.apply_dropout:apply_dropout}
        return self.sess.run(self.policy, feed_dict=feed_dict)
        
    def sampleAction(self, state, batch=1, apply_dropout=False):
        state = np.atleast_2d(state)
        policy = self._samplePolicy(state, apply_dropout)
        # Could either sample actions or take max action
        # For now take max
        action  = np.argmax(policy, axis=1)
        return action
    
    def samplePolicy(self, state, batch=32, apply_dropout=True):
        # import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        policy = self._samplePolicy(state, apply_dropout=apply_dropout)
    
        return policy

    
    def uncertainAction(self, state, training=True, batch=32):
        """
        Uses a Bayesian approach for getting uncertainty and an average action over
        a single state by using dropout at test times
        """
        # import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        policy = self._samplePolicy(state, training)
        # Could either sample actions or take max action
        # For now take max
        policy_avg = np.mean(policy, axis=0, keepdims=True)
        policy_std = np.std(policy, axis=0)
        
        action  = np.argmax(policy_avg)
        action_var = policy_std[action]
        
        avg_var = np.zeros((policy.shape[1], policy.shape[1]))
        for i in range(policy.shape[0]):
            avg_var += np.dot(policy[None, i, :].T, policy[None, i, :])
        avg_var = avg_var / float(policy.shape[0])
        exp_var = np.dot(policy_avg.T, policy_avg)
        
        predictive_variance = avg_var - exp_var
        
        return action, action_var
    
    def QBCAction(self, state, training=True, batch=32):
        """
        Uses query by committee to select the next action
        """
        # import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        policy = self._samplePolicy(state, training)
        # Could either sample actions or take max action
        # For now take max
        all_actions = np.argmax(policy, axis=1)
        action, _ = stats.mode(all_actions)
        action = action[0]
        disagree = 0
        for act in all_actions:
            if act != action: 
                disagree +=1
        
        return action, disagree        
        
    def save_model(self, expert_samples=-1):
        savefile = os.path.join(self.filepath, 'checkpoints/model-'+ str(expert_samples) + '_samples.ckpt')
        self.saver.save(self.sess, savefile)
        # self.saver.save(self.sess, savefile, global_step=tf.train.global_step(self.sess, self.global_step))
        print('Saved Model as {}'.format(savefile))
        
        
        
        
        
        
        
        

        