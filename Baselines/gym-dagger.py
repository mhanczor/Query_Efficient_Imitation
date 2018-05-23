import sys, argparse
import numpy as np
import tensorflow as tf
import gym
from dagger import DAgger
from efficient_dagger import Efficient_DAgger
from experts import CartPole_SubExpert, CartPole_iLQR, LunarLander_Expert

import matplotlib

# Build a learner model

class CartPoleAgent(object):
    
    def __init__(self, sess, env, lr=0.001, filepath='tmp/'):
        
        self.sess = sess
        input_size = (None,) + env.observation_space.shape
        output_size = env.action_space.n
        dropout_rate = 0.2
        
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
        # import pdb; pdb.set_trace()
        state = np.atleast_2d(state)
        state = np.repeat(state, batch, axis=0)
        feed_dict = {self.state : state, self.training : training}
        policy = self.sess.run(self.policy, feed_dict=feed_dict)
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
    
    def update(self, batch):
        state = batch[:, :-1]
        expert_action = batch[:,-1]
        feed_dict = {self.state: state, self.expert_action: expert_action, self.training : True}
        _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
        return loss
        
    def save_model_weights(self, filepath):
        self.saver.save(self.sess, filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        
def plot(stats):
    pass
    
        
def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-config-path', dest='model_config_path',
    #                     type=str, default='LunarLander-v2-config.json',
    #                     help="Path to the model config file.")
    parser.add_argument('--episodes', dest='num_episodes', type=int,
                        default=10, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser_group = parser.add_mutually_exclusive_group(required=False)
    # parser_group.add_argument('--render', dest='render',
    #                           action='store_true',
    #                           help="Whether to render the environment.")
    # parser_group.add_argument('--no-render', dest='render',
    #                           action='store_false',
    #                           help="Whether to render the environment.")
    # parser.set_defaults(render=False)
    
    parser.add_argument('--env', dest='env_name',
                              type=str, default='CartPole-v1',
                              help="Environment Name")
    
    parser.add_argument('--file', dest='file_name',
                              type=str, default='-1',
                              help="Filename to save.")
                              
    parser.add_argument('--load', dest='load_model',
                              action='store_true',
                              help="Whether to load the model.")
    parser.set_defaults(load=False)
    
    parser.add_argument('--thresh', dest='uncertainty_threshold', type=float,
                        default=0.1, help="Threshold for efficient dagger.")
    
    parser.add_argument('--loadfile', dest='load_file', type=str,
                              default='',
                              help="Load model")
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    episodes = args.num_episodes
    filename = args.file_name
    env_name = args.env_name
    var_thresh = args.uncertainty_threshold
        
    filepath = 'experiments/' + env_name + '/'+filename+'/'
    
    env = gym.make(env_name)
    sess = tf.Session()
    learner = CartPoleAgent(sess, env, lr=0.001, filepath=filepath)
    
    # expert = CartPole_SubExpert()
    # expert = CartPole_iLQR(env.env)
    expert = LunarLander_Expert()
    
    # dagger = DAgger(env, learner, expert, mixing=0.0)
    dagger = Efficient_DAgger(env, learner, expert, mixing=0.0, certainty_thresh=var_thresh)
    
    
    rewards, stats = dagger.trainAgent(episodes=episodes, mixing_decay=1.0)
    
    with open(filepath + 'stats.csv', 'a') as f:
        for line in stats:
            line = map(str, line)
            f.write(', '.join(line) + '\n')
    
    for i in range(5):
        # dagger.runEpisode(expert, render=True)
        dagger.runEpisode(learner, render=True)
    

if __name__ == "__main__":
    main(sys.argv)
        
        