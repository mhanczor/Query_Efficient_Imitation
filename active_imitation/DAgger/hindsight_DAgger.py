from __future__ import print_function
import numpy as np
import tensorflow as tf

from dagger import DAgger
# from DeepQ import DQNetwork
import random


class Hindsight_DAgger(DAgger):
    
    def __init__(self, env, learner, expert, mixing=0.0, random_samp=False):
        super(Hindsight_DAgger, self).__init__(env, learner, expert, mixing)
        self.random_samp = random_samp
        # self.learner_predict = self.entropyAction
        self.learner_predict = self.QBC_KL
        
    def entropyAction(self, state):
        # select an action return the entropy, over a single sample or multiple
        policy = self.learner.samplePolicy(state)
        entropy = -np.dot(policy, np.log(policy).T)
        ## When cast to np.float64 this can cause errors in np multinomial
        ## Normalize if this is larger than 1 due to rounding
        policy.astype(np.float64)
        pol_sum = np.sum(policy)
        if pol_sum > 1.0:
            # import pdb; pdb.set_trace()
            policy = policy / pol_sum
        try:        
            action = np.random.multinomial(1, policy[0])
            action = np.argmax(action)
        except:
            action = np.argmax(policy[0])

        return action, entropy
    
    def QBC_KL(self, state):
        """
        Uses multiple passes through a dropout NN as an approximation for
        multiple hypotheses sampled from a Gaussian
        
        Calculates the KL divergence between the committee members
        """
        #TODO clean this up so there is no concern with shape casting, make it so that
        # for arbitrary number of committee members and state space this works
        
        sample = self.learner.dropoutSample(state, batch=32)
    
        # need the average probability of each action
        # The consensus probability, average probability of each action
        p_consensus = np.sum(sample, axis=0)/sample.shape[0]
        log_diff = np.log(sample) - np.log(p_consensus)
        
        # Compute the KL divergence of each committee member to the consensus probability
        kl_div = np.sum(log_diff * sample, axis=1)
        # Average over the committee members to get the average divergence
        avg_kl = np.sum(kl_div) / sample.shape[0]
        
        policy_sum = np.sum(p_consensus)
        if policy_sum > 1.0:
            p_consensus = p_consensus / policy_sum
        # import pdb; pdb.set_trace()
        action = np.argmax(p_consensus)
        # try:
        #     action = np.random.multinomial(1, p_consensus)
        #     action = np.argmax(action)
        # except:
        #     action = np.argmax(policy)
            
        # Return the average divergence along with the sampled action from the consensus        
        return action, avg_kl
    
    def varianceAction(self, state):
        # select an action based on multiple forward passes through the network
        # return the variance over the selected action, which is the argmax of the average over passes
        action, action_var = self.learner.uncertainAction(state, batch=32)
        return action, action_var   
    
    def generateExpertSamples(self, mixing_decay=0.1, num_samples=1, save_image=False):
        """
        Information theoretic approach to selecting samples in hindsight
        
        num_samples - number of expert samples to collect per episode
        """
        state = self.env.reset()
        done = False
        trajectory = []
        expert_samples = 0
        img_arr = []
        
        if save_image: # For saving the image of individual states to see when the expert is queried
            unwr_env = self.env.unwrapped
        
        # import ipdb; ipdb.set_trace()
        # Generate an episode, store all the visited states
        while not done:
            learner_action, action_metric = self.learner_predict(state)
            
            mixing_prob = random.random()
            if mixing_prob >= self.mixing:
                action = self.expert.selectAction(state)
                store = state.tolist()
                store.append(action)
                self.dataset.append(store)
                expert_samples += 1
            else:
                action = learner_action
                traj_store = state.tolist()
                traj_store.append(action_metric)
                trajectory.append(traj_store)
            
            if save_image:
                arr = self.env.render(mode='rgb_array')
                img_arr.append(arr)
                
            state, reward, done, _ = self.env.step(action)
        
        if len(trajectory) != 0:
            """
            select samples, TODO change this to be able to collect multiple samples
            samples are the max over the action metric, this metric can be whatever
            """
            trajectory = np.array(trajectory)
            if self.random_samp:
                best_ind = random.randint(0, len(trajectory)-1)
            else:
                best_ind = np.argmax(trajectory[:,-1])
            state = trajectory[best_ind, :-1]
            action = self.expert.selectAction(state)
            store = state.tolist()
            store.append(action)
            self.dataset.append(store)
            expert_samples += 1
            
            # if num_samples == 1:
            #     sample_metric  = tf.Summary(value=[tf.Summary.Value(tag='Selected_Sample_Metric_Value', simple_value=valid_reward)])
            #     self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
            
            state_imgs = []
            if save_image:
                state_imgs = [img_arr[best_ind]]
        
        # After the episode ends select num_samples samples to get expert labelling for
        # How to collect the n most needed points to label?
        # What is the criteria for datapoint selection?
        
        # Could be uncertainty over the states?
        # Could be expected variance reduction

        return expert_samples, state_imgs #TODO do we actually need to return any of this?
    