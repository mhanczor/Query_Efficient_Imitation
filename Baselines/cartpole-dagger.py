import numpy as np
import tensorflow as tf
import gym
from dagger import DAgger
from experts import CartPole_SubExpert, CartPole_iLQR


# Build a learner model

class CartPoleAgent(object):
    
    def __init__(self, sess, env, lr=0.001, filepath='tmp/'):
        
        self.sess = sess
        input_size = (None,) + env.observation_space.shape
        output_size = env.action_space.n
        dropout_rate = 0.1
        
        with tf.name_scope("Inputs"):
            self.state = tf.placeholder(tf.float32, input_size, name='State')
            self.expert_action = tf.placeholder(tf.int32, [None], name='Expert_Action')
            self.training = tf.placeholder(tf.bool)
        with tf.name_scope("Model"):
            fc_1 = tf.layers.dense(inputs=self.state, units=16, activation=tf.nn.relu)
            dropout_1 = tf.layers.dropout(inputs=fc_1, rate=dropout_rate, training=self.training)
            fc_2 = tf.layers.dense(inputs=dropout_1, units=16, activation=tf.nn.relu)
            dropout_2 = tf.layers.dropout(inputs=fc_2, rate=dropout_rate, training=self.training)
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
    
    def selectAction(self, state, training=False):
        state = np.atleast_2d(state)
        feed_dict = {self.state : state, self.training : training}
        policy = self.sess.run(self.policy, feed_dict=feed_dict)
        # Could either sample actions or take max action
        # For now take max
        action  = np.argmax(policy)
        return action
    
    def uncertainAction(self, state, training=True, batch=32):
        """
        Uses a Bayesian approach for getting uncertainty and an average action over
        a single state by using dropout at test times
        """
        import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        feed_dict = {self.state : state, self.training : training}
        policy = self.sess.run(self.policy, feed_dict=feed_dict)
        # Could either sample actions or take max action
        # For now take max
        policy_avg = np.mean(policy, axis=0)
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
    
    def update(self, batch):
        state = batch[:, :-1]
        expert_action = batch[:,-1]
        feed_dict = {self.state: state, self.expert_action: expert_action, self.training : True}
        _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
        return loss
        
    def save_model_weights(self, filepath):
        self.saver.save(self.sess, filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        
        

def main():
    episodes = 10
    
    env = gym.make('CartPole-v1')
    sess = tf.Session()
    learner = CartPoleAgent(sess, env, lr=0.001, filepath='tmp/CP-DAgger/')
    # expert = CartPole_SubExpert()
    expert = CartPole_iLQR(env.env)
    # dagger = DAgger(env, learner, expert, mixing=0.0)
    dagger = Efficient_DAgger(env, learner, expert, mixing=0.0)
    
    rewards = dagger.trainAgent(episodes=episodes, mixing_decay=1.0)
    
    for i in range(10):
        # dagger.runEpisode(expert, render=True)
        dagger.runEpisode(learner, render=True)
    

if __name__ == "__main__":
    main()
        
        