from __future__ import print_function
import numpy as np

from active_imitation.learners import DAgger
import random


class Hindsight_DAgger(DAgger):
    
    def __init__(self, env, learner, expert, agg_buffer, action_selection, 
                    mixing=0.0, random_sample=False, continuous=False):
        super(Hindsight_DAgger, self).__init__(env, learner, expert, agg_buffer, mixing, continuous)
        self.random_sample = random_sample
        self.learner_predict = action_selection # This is a function 
    
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
        
        if save_image: # For saving the image of individual states to see when the expert is queried
            img_arr = []
            unwr_env = self.env.unwrapped
        
        # Generate an episode, store all the visited states
        while not done:
            learner_action, action_uncertainty = self.learner_predict(self.learner, state)
            mixing_prob = random.random()
            if mixing_prob >= self.mixing:
                action = self.expert.sampleAction(state)
                self.dataset.store(state, action)
                expert_samples += 1
            else:
                action = learner_action.squeeze()
                trajectory_belief.append(action_uncertainty)
                trajectory_states.append(state)
            
            if save_image:
                arr = self.env.render(mode='rgb_array')
                img_arr.append(arr)
            
            try:
                state, reward, done, _ = self.env.step(action)
            except:
                print('Hingsight Generate \n\
                        State: {} \n Action: {}'.format(state, action))
                import pdb; pdb.set_trace()
        
        if len(trajectory_belief) > 0:
            """
            select samples, TODO change this to be able to collect multiple samples
            samples are the max over the action metric, this metric can be whatever
            """
            trajectory_belief = np.array(trajectory_belief)
            if self.random_sample:
                best_ind = random.randint(0, len(trajectory_belief)-1)
            else:
                # Select the state that had the highest uncertainty
                # If there are multiple states with the same uncertainty, randomly select between them
                trajectory_belief = trajectory_belief.squeeze()
                best_val = np.max(trajectory_belief)
                val_ind = np.argwhere(trajectory_belief == best_val)
                if val_ind.shape[0] > 1:
                    print('WARNING: SELECTING FROM MULTIPLE EQUIVALENT STATES')
                    best_ind = np.random.choice(val_ind.squeeze())
                    print(best_val)
                else:
                    best_ind = val_ind[0,0]

            state = trajectory_states[best_ind]
            action = self.expert.sampleAction(state)
            self.dataset.store(state, action)
            expert_samples += 1
            
            # if num_samples == 1:
            #     sample_metric  = tf.Summary(value=[tf.Summary.Value(tag='Selected_Sample_Metric_Value', simple_value=valid_reward)])
            #     self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
            selected_utility = trajectory_belief[best_ind]
        else:
            selected_utility = 0. # Don't select examples if the expert is providing them all 
        
        self.mixing += mixing_decay    
        state_imgs = []
        if save_image:
            state_imgs = [img_arr[best_ind]]

        return expert_samples, state_imgs, selected_utility #TODO do we actually need to return any of this?
    
    
    
    
    
    