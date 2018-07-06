from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

class DQNetwork(object):
    
    # Deep Q network for solving environments MountainCar and CartPole
    # Take in state information as an input, output q-value for each action
    def __init__(self, environment, sess, alpha=0.0001, filepath='tmp/deepq/', is_dueling=False, is_target=False):

        
        self.sess = sess
        env = environment
        self.nA = env.action_space.n
        self.nObs = (None,) + env.observation_space.shape
        self.filepath = filepath
        self.is_target = is_target
        
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.float32, self.nObs, name='Features')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
            self.action = tf.placeholder(tf.int32, [None], name='Selected_Action')
        with tf.name_scope("Layers"):
            fc_1 = tf.layers.dense(inputs=self.x, units=32, activation=tf.nn.relu)
            fc_2 = tf.layers.dense(inputs=fc_1, units=32, activation=tf.nn.relu)
        with tf.name_scope("Output"):
            
            if is_dueling:
                # Dueling DQN
                advantage_dense = tf.layers.dense(inputs=fc_2, units=16, activation=tf.nn.relu)
                advantage_stream =tf.layers.dense(inputs=advantage_dense, units=self.nA)
                self.advantage = tf.subtract(advantage_stream, tf.reduce_mean(advantage_stream))
                value_dense = tf.layers.dense(inputs=fc_2, units=16, activation=tf.nn.relu)
                value = tf.layers.dense(inputs=value_dense, units=1)
                self.q_pred = tf.add(value, self.advantage, name='Q_predicted')
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True)
            else:
                # Vanilla DQN
                fc_3 = tf.layers.dense(inputs=self.x, units=64, activation=tf.nn.relu) # Changed this to make the network a single hidden layer
                self.q_pred = tf.layers.dense(inputs=fc_3, units=self.nA, name='Q_predicted')
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True) # Qval for the chosen action, should have a dimension (None, 1)
        import pdb; pdb.set_trace()
        with tf.name_scope("Loss"):
            if is_dueling:
                regularizer = 0.01*(tf.nn.l2_loss(fc_1) + tf.nn.l2_loss(fc_2) + tf.nn.l2_loss(advantage_dense) + tf.nn.l2_loss(value_dense))
            else:
                regularizer = 0.01*(tf.nn.l2_loss(fc_1) + tf.nn.l2_loss(fc_2) + tf.nn.l2_loss(fc_3))
            self.loss = tf.losses.huber_loss(self.q_target, self.q_action) + regularizer
            self.loss_summary = tf.summary.scalar("Loss", self.loss)
        with tf.name_scope("Optimize"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # self.opt = tf.train.AdamOptimizer(alpha).minimize(self.loss, global_step=self.global_step)
            train_opt = tf.train.AdamOptimizer(alpha)
            grad_norm = 10
            if grad_norm != None:
                grads_and_vars = train_opt.compute_gradients(self.loss)
                for idx, (grad, var) in enumerate(grads_and_vars):
                  if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, grad_norm), var)
                self.opt = train_opt.apply_gradients(grads_and_vars, global_step=self.global_step)
            else:
                self.opt = train_opt.minimize(self.loss, global_step=self.global_step)
                
        self._reset()
    
    def _reset(self):
        self.sess.run(tf.global_variables_initializer())
        if not self.is_target:
            self.saver = tf.train.Saver(max_to_keep=10)
            self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
            # self.builder = tf.saved_model.builder.SavedModelBuilder(self.filepath+'models/')
        else:
            pass
        
    def infer(self, features):
        # Evaluate the data using the model
        feed_dict = {self.x: features} # Features is a (batch, obs_space) matrix
        q_vals = self.sess.run(self.q_pred, feed_dict=feed_dict)
        return q_vals
        
    def update(self, features, q_target, action=None):
        # Update the model by calculating the loss over a selected action
        # import pdb; pdb.set_trace()
        action = action.flatten() # Actions must be in a 1d aray
        feed_dict = {self.x: features, self.action: action, self.q_target: q_target}
        _, loss_summary, loss, onehot, act, pred, target, q_act = self.sess.run([self.opt, self.loss_summary, self.loss, self.q_onehot, self.action, self.q_pred, self.q_target, self.q_action], feed_dict=feed_dict)
        return loss_summary, loss
    
    def targetGraphUpdate(self):
        # As of now just straight copying the current to the target, can add some rate later if needed
        variables = tf.trainable_variables()
        net_vars = len(variables) // 2 # Since we have two graphs we just want the first graph vars to update the second
        ops = []
        for idx, var in enumerate(variables[:net_vars]):
            value_op = variables[idx+net_vars].assign(value=var.value())
            self.sess.run(value_op)
        print('Updated Target Network')
        
    def getFeatures(self, S):
        # Used here to make agent compatible with multiple state information types
        return np.atleast_2d(S)
        
    def save_model(self):
        tf.saved_model.simple_save(self.sess,
                                    export_dir=self.filepath + 'saved_model/',
                                    inputs={'x':self.x},
                                    outputs={'y':self.q_pred})
        print("Saved Model")
            
    def save_model_weights(self):
        # Helper function to save your model / weights. 
        self.saver.save(self.sess, self.filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        print("Saved Weights")
        
    def load_model(self, model_file):
        # If needed
        raise NotImplementedError

    def load_model_weights(self, weight_file=''):
        # Helper funciton to load model weights.
        if weight_file == '':
            filename = self.filepath+'checkpoints/'
        else:
            filename = weight_file
        
        latest_ckpt = tf.train.latest_checkpoint(filename)
        if latest_ckpt:
            self.saver.restore(self.sess, latest_ckpt)
            print('Loaded weights from {}'.format(latest_ckpt))
        elif weight_file != '':
            try:
                self.saver.restore(self.sess, filename)
                print('Loaded weights from {}'.format(filename))
            except:
                print("Loading didn't work")
        else:
            print('No weight file to load, starting from scratch')
            return -1
        
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
        
        