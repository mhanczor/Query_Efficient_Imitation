import numpy as np
import tensorflow as tf

from dagger import DAgger
# from DeepQ import DQNetwork
import random


"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
"""


class Efficient_DAgger(DAgger):
    
    def __init__(self, env, learner, expert, mixing=0.0, certainty_thresh=0.1):
        super(Efficient_DAgger, self).__init__(env, learner, expert, mixing)
        self.thresh = certainty_thresh
        
    
    def generateExpertSamples(self, mixing_decay=0.1):
        
        # Initialize environment and run a mixed trajectory
        # import pdb; pdb.set_trace()
        total_reward = 0
        state = self.env.reset()
        done = False
        expert_samples = 0
        while not done:
            """
            Only get expert actions when the learner is uncertain about the current state
            Reduces the amount of times the expert is queried
            
            Could still use the learner in the uncertain areas but see what the expert would have done?
            """
            uncer_action, action_var = self.learner.uncertainAction(state)
            # print(action_var)
            # import pdb; pdb.set_trace()  
            mixing_prob = random.random()
            if action_var >= self.thresh or mixing_prob >= self.mixing: 
                if mixing_prob >= self.mixing:
                    action = self.expert.selectAction(state)
                    expert_action = action
                else:
                    action = uncer_action
                    expert_action = self.expert.selectAction(state)
                store = state.tolist()
                store.append(expert_action)
                self.dataset.append(store)
                expert_samples += 1
            else:
                action = uncer_action
            
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        
        self.mixing += mixing_decay
        if self.mixing > 1.0: self.mixing = 1.0
                
        return state, reward, done, expert_samples
        


if __name__ == "__main__":
    pass
    