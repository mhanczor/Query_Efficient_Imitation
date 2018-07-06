from __future__ import absolute_import, division, print_function

from gym.envs.classic_control import rendering # Have to impor this before tensorflow, needs to be removed when using AWS

import tensorflow as tf
import numpy as np
from DeepQ import DQNetwork
import gym, sys, copy, argparse, time, os
import random

class DQN_Agent(object):
    
    def __init__(self, environment, sess, network_type, render=False, gamma=1., alpha=0.001, filepath='tmp/', double=False):

        self.env = environment
        self.nA = self.env.action_space.n
        self.render = render
        self.gamma = gamma
        self.is_double = False
                
        # if network_type == 'Linear':
        #     self.net = LinearQ(environment, sess=sess, filepath=filepath, alpha=alpha)
        #     self.linear = True
        if network_type == 'DNN':
            self.net = DQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha)
            self.linear = False
        elif network_type == 'DDNN':
            self.net = DQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha, is_dueling=True)
            if double == True:
                self.target_net = DQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha, is_dueling=True, is_target=True)
                self.is_double = True
            self.linear = False
        # elif network_type == 'DCNN':
        #     self.net = ConvQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha)
        #     if double == True:
        #         self.target_net = ConvQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha, is_target=True)
        #         self.is_double = True
        #     self.linear = False
        else:
            raise ValueError

    def train(self, episodes=1e3, epsilon=0.7, decay_rate=4.5e-6, replay=False, check_rate=1e4, memory_size=50000, burn_in=10000):
        # Interact with the environment and update the model parameters
        # If using experience replay then update the model with a sampled minibatch
        if replay: # If using experience replay, need to burn in a set of transitions
            if self.linear:
                memory_queue = Replay_Memory()
                self.burn_in_memory(memory_queue, burn_in=10000)
                batch_size = 32
                print('Memory Burned In')
            else:
                memory_queue = Replay_Memory(memory_size=memory_size)
                self.burn_in_memory(memory_queue, burn_in=burn_in)
                batch_size = 32
                print('Memory Burned In')
        else:
            batch_size = 1
                
        iters = 0
        test_reward = 0
        reward_summary = tf.Summary()
        ep_reward_summary = tf.Summary()
        avg_action_q_pred = tf.Summary()
        # pdb.set_trace()
        for ep in range(int(episodes)):
            ep_reward = 0
            ep_iters = 0
            avg_episode_q = 0
            S = self.env.reset()
            if not self.linear:
                for i in range(4):
                    features = self.net.getFeatures(S) # Fill the Atari buffer with frames
            done = False
            partial_episode=False
            while not done:
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < epsilon:
                    action = np.random.randint(self.nA)
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                ep_reward += R
                if not replay:
                    if done:
                        q_target = np.array([[R]])
                    else:
                        feature_next = self.net.getFeatures(S_next)
                        if self.is_double:
                            act_2 = np.argmax(self.net.infer(feature_next)) # Need to evaluate the greedy policy based on the current net, not the target
                            q_vals_next = self.target_net.infer(feature_next)
                            q_target = np.array([self.gamma*q_vals_next[:,act_2] + R])
                        else:
                            q_vals_next = self.net.infer(feature_next)
                            q_target = np.array([[self.gamma*np.max(q_vals_next) + R]])
                        
                    if self.linear:
                        features = features[None,action,:]
                        
                    avg_episode_q += np.max(q_vals_next)
                    pdb.set_trace()
                    summary, loss = self.net.update(features, q_target, action=np.array([[action]])) 
                    
                    if np.isnan(loss):
                        print("Loss exploded")
                        return          
                    
                    self.net.writer.add_summary(summary, tf.train.global_step(self.net.sess, self.net.global_step))
                else:
                    # Update the gradient with experience replay
                    if self.linear:
                        features = features[None,action,:]
                    feature_next = self.net.getFeatures(S_next)
                    store = (features, action, R/500.0, done, feature_next) # TODO added this divs by 500 to bound rewards
                    # Store the tuple (features, action, R, done, feature_next)
                    memory_queue.append(store)
                    
                    # Ranomly select a batch of tuples from memory
                    cur_features, actions, rewards, dones, next_features = memory_queue.sample_batch(batch_size=batch_size, is_linear=self.linear)
                    
                    # Need to differentiate between features for linear and deep models
                    # Features for the linear model are state and action dependent
                    if self.linear:
                        best_q = np.zeros((batch_size,1))
                        for i, ele in enumerate(next_features):
                            best_q[i,:] = np.max(self.net.infer(ele))
                    else:
                        if self.is_double:
                            act_2 = np.argmax(self.net.infer(next_features), axis=1) # Need to evaluate the greedy policy based on the current net, not the target
                            q_vals_next = self.target_net.infer(next_features)
                            act_index = [np.arange(act_2.shape[0]), act_2, None]
                            # import pdb; pdb.set_trace()
                            best_q = q_vals_next[act_index]
                        else:
                            best_q = self.net.infer(next_features)
                            best_q = np.max(best_q, axis=1, keepdims=True)
                    avg_episode_q += np.sum(best_q)                                        
                    done_mask = 1 - dones.astype(int) # Makes a mask of 0 where done is true, 1 otherwise
                    q_target = self.gamma*best_q * done_mask + rewards # If done, target just reward, else target reward + best_q
                    
                    # Update the gradients with the batch
                    summary, loss = self.net.update(cur_features, q_target, action=actions)
                    if np.isnan(loss):
                        print("Loss exploded")
                        return
                    self.net.writer.add_summary(summary, tf.train.global_step(self.net.sess, self.net.global_step))
                    
                S = S_next # Update the state info
                if epsilon > 0.1: # Keep some exploration
                    epsilon -= decay_rate # Reduce epsilon as policy learns
                
                if iters % check_rate == 0:
                    # Test the model performance
                    test_reward, test_rewards = self.test(episodes=5, epsilon=0.05) # Run a test to check the performance of the model
                    test_rewards = np.array(test_rewards)
                    std_dev = np.std(test_rewards)
                    mean = np.mean(test_rewards)
                    print('Reward: {}, StdDev: {}, Step: {}'.format(test_reward, std_dev, tf.train.global_step(self.net.sess, self.net.global_step)))
                    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Test_Reward', simple_value=test_reward)])
                    self.net.writer.add_summary(reward_summary, tf.train.global_step(self.net.sess, self.net.global_step))
                    if self.is_double and iters != 0:
                        self.target_net.targetGraphUpdate() # Update the weights of the target graph
                    done = True
                    partial_episode = True
                iters += 1
                ep_iters += 1
            
            if not partial_episode:
                ep_reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode_Reward', simple_value=ep_reward)])
                self.net.writer.add_summary(ep_reward_summary, tf.train.global_step(self.net.sess, self.net.global_step))
                avg_episode_q = avg_episode_q / (batch_size*ep_iters)
                avg_action_q_pred  = tf.Summary(value=[tf.Summary.Value(tag='Episode_Avg_Q', simple_value=avg_episode_q)])
                self.net.writer.add_summary(avg_action_q_pred, tf.train.global_step(self.net.sess, self.net.global_step))
            if ep % 100 == 0:
                print("episode {} complete, epsilon={}".format(ep, epsilon))
                if self.is_double and iters != 0:
                    self.target_net.targetGraphUpdate() # Update the weights of the target graph
            if ep % 100 == 0  and ep != 0:
                self.net.save_model_weights()
        self.net.save_model()

    def test(self, model_file=None, episodes=100, epsilon=0.0):
        # Evaluate the performance of the agent over episodes
        total_reward = 0
        rewards = []
        for ep in range(int(episodes)):
            episode_reward = 0
            S = self.env.reset()
            if self.render:
                self.env.render()
            done = False
            while not done:
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < epsilon:
                    action = np.random.choice(self.nA)
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                episode_reward += R
                if self.render:
                    self.env.render()
                S = S_next # Update the state
            total_reward += episode_reward
            rewards.append(episode_reward)
        
        average_reward = float(total_reward/episodes)
        return average_reward, rewards
    
    def burn_in_memory(self, memory_queue, burn_in=10000):
        # Initialize the replay memory with a burn_in number of episodes / transitions. 
        i = 0
        while True:
            S = self.env.reset()
            done = False
            while not done:
                if i >= burn_in:
                    return
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < 0.5: # Add some stochasticity to the burn in
                    action = self.env.action_space.sample()
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                if self.linear:
                    features = features[None,action,:] # (1, state*action)
                feature_next = self.net.getFeatures(S_next)
                store = (features, action, R, done, feature_next)
                memory_queue.append(store)
                S = S_next
                i += 1

class Replay_Memory(object):

    def __init__(self, memory_size=50000):

        # The memory stores transitions from the agent
        # taking actions in the environment.
        self.memory = []
        self.memory_size = memory_size
        self.feature_shape = None

    def sample_batch(self, batch_size=32, is_linear=True):    
        # Return the data in matrix forms for easy feeding into networks
        # Data should be in in form (batch, values)
        if batch_size < 1 or batch_size >= self.size():
            raise ValueError
        
        batch = random.sample(self.memory, batch_size)
        batch_features = (batch_size,) + self.feature_shape[1:]
        cur_features = np.zeros(batch_features)
        actions = np.zeros((batch_size, 1), dtype=np.uint8)
        rewards = np.zeros((batch_size, 1), dtype=np.int32)
        dones = np.zeros((batch_size, 1), dtype=np.bool)
        if is_linear:
            next_features = []
        else:
            next_features = np.zeros(batch_features)
        
        for i, ele in enumerate(batch):
            cur_features[i,:] = ele[0]
            actions[i,:] = ele[1]
            rewards[i,:] = ele[2]
            dones[i,:] = ele[3]
            if is_linear:
                next_features += (ele[4],) # For linear state and action features
            else:
                next_features[i,:] = ele[4]   
        return (cur_features, actions, rewards, dones, next_features)

    def append(self, transition):
        
        if not self.feature_shape: #On the first entry record the shape of the features
            self.feature_shape = transition[0].shape
                    
        # Appends transition to the memory.
        if self.size() >= self.memory_size:
              self.memory.pop()
        self.memory.insert(0,transition)
        
        if self.size() > self.memory_size:
            print('Queue Overfilled')
        
    def size(self):
        return len(self.memory)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='CartPole-v1')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--load',dest='load',type=int,default=0)
    parser.add_argument('--model',dest='model_file',type=str, default='')
    parser.add_argument('--network',dest='network',type=str, default='Linear')
    parser.add_argument('--path', dest='filepath', type=str, default='tmp/')
    parser.add_argument('--replay',dest='replay',type=int,default=0)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    network_type = args.network
    train = args.train
    load = args.load
    filepath = args.filepath
    replay = args.replay
    model_file = args.model_file

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    
    # Setting the session to allow growth, so it doesn't allocate all GPU memory
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # DeepQ Network
    episodes= 500 #1e3
    epsilon=0.5
    gamma = 0.99
    alpha = 0.0001
    network = 'DDNN' # Deep network, not dueling
    replay = True
    double = False 

    # Create an instance of the environment and agent
    env = gym.make(environment_name)
    
    if filepath[-1] != '/':
        filepath += '/'
    
    # Initialize agent
    agent = DQN_Agent(environment=env, 
                        sess=sess, 
                        network_type=network,
                        gamma=gamma,
                        filepath=filepath,
                        alpha=alpha,
                        double=double)
    if load:
        if model_file:
            agent.net.load_model_weights(weight_file=model_file)
        else:
            agent.net.load_model_weights()
    
    if train:    
        # Train the network
        agent.train(episodes=episodes,
                    epsilon=epsilon,
                    replay=replay,
                    check_rate=1000)
        agent.net.save_model_weights()

        total_reward, rewards = agent.test(episodes=100, epsilon=0.00)
        # print("Tested total average reward: {}".format(total_reward))
        rewards = np.array(rewards)
        std_dev = np.std(rewards)
        mean = np.mean(rewards)
        print("Training Mean: {}, StdDev: {}".format(mean, std_dev))
    else:
        agent.render = True
        agent.test(episodes=10, epsilon=0.00)
        agent.render = False
        total_reward, rewards = agent.test(episodes=100, epsilon=0.00)
        # print("Tested total average reward: {}".format(total_reward))
        rewards = np.array(rewards)
        std_dev = np.std(rewards)
        mean = np.mean(rewards)
        print("Test Mean: {}, StdDev: {}".format(mean, std_dev))
    
    agent.net.writer.close()    
    env.close()
    sess.close()
    

if __name__ == '__main__':
    main(sys.argv)
