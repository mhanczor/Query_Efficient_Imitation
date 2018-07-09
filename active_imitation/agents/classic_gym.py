import numpy as np
import tensorflow as tf
from scipy import stats

#TODO:
#   Change env to only pass in the env dimensions required

class GymAgent(object):
    
    def __init__(self, sess, env, lr=0.001, dropout_rate=0.1, filepath='tmp/'):
        """
        Learner agent for OpenAI Gym's classic environments like CartPole and LunarLander
        
        Args:
            sess(tf session): current tensorflow session
            env (gym_env): OpenAI gym environment (WILL CHANGE!)
            lr (float): learning rate for the network
            dropout_rate[float]: Probability of dropout for any node, in range [0,1],
                                a value of 0 would lead to no dropout
            filepath[str]: policy and data save location        
        """
        
        self.sess = sess
        input_size = (None,) + env.observation_space.shape
        output_size = env.action_space.n
        
        with tf.name_scope("Inputs"):
            self.state = tf.placeholder(tf.float32, input_size, name='State')
            self.expert_action = tf.placeholder(tf.int32, [None], name='Expert_Action')
            self.apply_dropout = tf.placeholder(tf.bool)
        with tf.name_scope("Model"):
            fc_1 = tf.layers.dense(inputs=self.state, units=16, activation=tf.nn.relu)
            dropout_1 = tf.layers.dropout(inputs=fc_1, rate=dropout_rate, training=self.apply_dropout)
            fc_2 = tf.layers.dense(inputs=dropout_1, units=16, activation=tf.nn.relu)
            dropout_2 = tf.layers.dropout(inputs=fc_2, rate=dropout_rate, training=self.apply_dropout)
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
        state = batch[:, :-1]
        expert_action = batch[:,-1]
        feed_dict = {self.state: state, self.expert_action:expert_action, self.apply_dropout:True}
        _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
        return loss
    
    def samplePolicy(self, state, apply_dropout=False):
        state = np.atleast_2d(state)
        feed_dict = {self.state : state, self.apply_dropout:apply_dropout}
        return self.sess.run(self.policy, feed_dict=feed_dict)
        
    def sampleAction(self, state, batch=1, apply_dropout=False):
        state = np.atleast_2d(state)
        policy = self.samplePolicy(state, apply_dropout)
        # Could either sample actions or take max action
        # For now take max
        action  = np.argmax(policy)
        return action
    
    # def dropoutSample(self, state, batch=32):
    #     # import pdb; pdb.set_trace()
    #     state = np.atleast_2d(state)
    #     state = np.repeat(state, batch, axis=0)
    #     policy = self.samplePolicy(state, apply_dropout=True)
    # 
    #     return policy

    
    def uncertainAction(self, state, training=True, batch=32):
        """
        Uses a Bayesian approach for getting uncertainty and an average action over
        a single state by using dropout at test times
        """
        # import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        policy = self.samplePolicy(state, training)
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
        policy = self.samplePolicy(state, training)
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
        
    def save_model_weights(self, filepath):
        self.saver.save(self.sess, filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        