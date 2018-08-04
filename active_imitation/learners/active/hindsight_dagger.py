from __future__ import print_function
import numpy as np

from active_imitation.learners import DAgger
import random
import time


class Hindsight_DAgger(DAgger):
    
    def __init__(self, env, learner, expert, agg_buffer, action_selection, 
                    mixing=0.0, random_sample=False, continuous=False, density_weight=0.0, budget=1, **kwargs):
        super(Hindsight_DAgger, self).__init__(env, learner, expert, agg_buffer, mixing, continuous, **kwargs)
        self.random_sample = random_sample
        self.learner_predict = action_selection # This is a function 
        self.density_weight = density_weight
        self.budget = budget
    
    def generateExpertSamples(self, mixing_decay=0.1, num_samples=1, save_image=False):
        """
        Information theoretic approach to selecting samples in hindsight
        
        num_samples - number of expert samples to collect per episode
        """
        state = self.env.reset()
        done = False
        trajectory_belief = []
        trajectory_states = []
        expert_samples = 0
        states = []
        
        if save_image: # For saving the image of individual states to see when the expert is queried
            img_arr = []
            unwr_env = self.env.unwrapped
        
        # Generate an episode, store all the visited states
        successes = 0
        while not done:
            states.append(state)
            learner_action, action_uncertainty = self.learner_predict(self.learner, state)
            mixing_prob = random.random()
            if mixing_prob >= self.mixing:
                action = self.expert.sampleAction(state)
                self.dataset.store(state, action)
                expert_samples += 1
            else:
                action = learner_action
                trajectory_belief.append(action_uncertainty)
                trajectory_states.append(state)
            
            if save_image:
                arr = self.env.render(mode='rgb_array')
                img_arr.append(arr)
            
            state, reward, done, info = self.env.step(action)
            
            if self.continuous: # If using a robot env, check for success
                success = info['is_success']
                if success:
                    successes += 1
                else:
                    successes = 0
                if successes >= 5:
                    done = True
            # try:
            #     state, reward, done, _ = self.env.step(action)
            # except:
            #     print('Hingsight Generate \n\
            #             State: {} \n Action: {}'.format(state, action))
            #     import pdb; pdb.set_trace()
        
        if len(trajectory_belief) > 0:
            """
            select samples, TODO change this to be able to collect multiple samples
            samples are the max over the action metric, this metric can be whatever
            """
            trajectory_belief = np.array(trajectory_belief)
            
            # Apply density weighting to the states if necessary
            if self.density_weight:
                weighting = self.densityWeighting(states)**self.density_weight
                if self.random_sample:
                    trajectory_belief = weighting # If we just want points to be selected by density weighting and not by trajectory uncertainty
                else:
                    trajectory_belief *= weighting # Combining density weighting and uncertainty
                            
            N = len(trajectory_belief)
            if N <= self.budget:
                best_indices = np.arange(N) # If we have more budget than states, take them all 
            elif self.random_sample and not self.density_weight:
                best_indices = np.random.choice(N, self.budget) # If we are randomly selecting, return random indices up to the budget size
                # best_ind = random.randint(0, len(trajectory_belief)-1)
            else:
                # Use argpartition to quickly select the largest uncertainty indices unordered
                best_indices = np.argpartition(trajectory_belief, -self.budget)[-self.budget:]
        
            for index in best_indices:
                state = trajectory_states[index]
                action = self.expert.sampleAction(state)
                self.dataset.store(state, action)
                expert_samples += 1
            
            avg_selected_utility = np.mean(trajectory_belief[best_indices])
        else:
            avg_selected_utility = 0. # Don't select examples if the expert is providing them all 
        
        self.mixing += mixing_decay    
        state_imgs = []
        if save_image:
            state_imgs = [img_arr[best_ind]]

        return expert_samples, state_imgs, avg_selected_utility #TODO do we actually need to return any of this?
    
    # REMOVED THIS WARNING, SOMTHING TO THINK ABOUT!
    # best_val = np.max(trajectory_belief)
    # val_ind = np.argwhere(trajectory_belief == best_val)
    # if val_ind.shape[0] > 1:
    #     print('WARNING: SELECTING FROM MULTIPLE EQUIVALENT STATES')
    #     best_ind = np.random.choice(val_ind.squeeze())
    #     print(best_val)
    # else:
    #     best_ind = val_ind[0,0]
    
    def densityWeighting(self, obs):
        # For Fetch envs we'll only need the state, the goal is the same for every timestep
        # For gym, not sure what to do yet
        
        # Calculate the distance from each point to every other point
        N = (len(obs))
        if self.continuous:
            s_dim = obs[0]['observation'].shape[0]
            states = np.empty((N, s_dim))
            for i, state in enumerate(obs):
                states[i, :] = state['observation']
        else:
            # s_dim = obs[]
            states = np.array(obs)
            
        if states.ndim > 4:
            # Not really a good metric
            states = states.squeeze()
            avg_state = np.mean(states, axis=0, keepdims=True)
            density = np.mean(states - avg_state, axis=(1, 2, 3))
        else:
            density = np.empty((N))
            for i in range(N):
                norm = np.linalg.norm(states[i, :] - states, axis=1)
                avg = np.sum(norm)/N
                density[i] = avg
            density = 1./(density + 1e-3) #inverse distance, want more packed states ranked larger
        return density
        
    
    
    
    