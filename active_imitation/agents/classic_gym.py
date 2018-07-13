import numpy as np
import tensorflow as tf
from scipy import stats
import os

#TODO:
#   Change env to only pass in the env dimensions required


DEFAULT_PARAMS = {
    # 'layers': [16, 16, 16], # Layers and hidden units in network
    'lr': 0.001, # Learning rate
    'dropout_rate': 0.1, # Dropout rate during training and forward samples
    'filepath': '~/Research/experiments/tmp/'
}

class GymAgent(object):
    
    def __init__(self, env_dims, lr=0.001, dropout_rate=0.1, filepath='tmp/'):
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
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        self.dropout_rate = dropout_rate
        
        self.filepath = filepath
        input_size = (None,) + (env_dims['observation'],)
        output_size = env_dims['action_space']
        
        with tf.name_scope("Inputs"):
            self.state = tf.placeholder(tf.float32, input_size, name='State')
            self.expert_action = tf.placeholder(tf.int32, [None, ], name='Expert_Action')
            self.apply_dropout = tf.placeholder(tf.bool)
        with tf.name_scope("Model"):
            fc_1 = tf.layers.dense(inputs=self.state, units=16, activation=tf.nn.relu)
            dropout_1 = tf.layers.dropout(inputs=fc_1, rate=self.dropout_rate, training=self.apply_dropout)
            fc_2 = tf.layers.dense(inputs=dropout_1, units=16, activation=tf.nn.relu)
            dropout_2 = tf.layers.dropout(inputs=fc_2, rate=self.dropout_rate, training=self.apply_dropout)
            logits = tf.layers.dense(inputs=dropout_2, units=output_size, activation=None)
            self.policy = tf.nn.softmax(logits, name="Policy_Output")
        with tf.name_scope("Loss"):
            regularizer = tf.nn.l2_loss(fc_1) + tf.nn.l2_loss(fc_2) + tf.nn.l2_loss(logits)
            labels = tf.one_hot(self.expert_action, output_size, axis=-1)
            self.loss = tf.losses.softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.SUM) + \
                regularizer * 0.001
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss)
    
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(filepath+'events/', self.sess.graph)
        
    
    def update(self, batch):
        feed_dict = {self.state:batch['observation'], self.expert_action:batch['action'].flatten(), 
                    self.apply_dropout:True}
        _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
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
        action  = np.argmax(policy)
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
        
        
        
        
        
        
        
        

        