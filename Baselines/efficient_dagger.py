import numpy as np
import tensorflow as tf


# from DeepQ import DQNetwork
import random


"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
"""


class Efficient_DAgger(DAgger):
    
    
    def generateExpertSamples(self, certainty_thresh=0.1, mixing_decay=0.1):
        
        # Initialize environment and run a mixed trajectory
        import pdb; pdb.set_trace()
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            """
            Only get expert actions when the learner is uncertain about the current state
            Reduces the amount of times the expert is queried
            """
            uncer_action, action_var = self.learner.uncertainAction(state)
            if action_var >= certainty_thresh:
                action = self.expert.selectAction(state)
                expert_action = action
                store = state.tolist()
                store.append(expert_action)
                self.dataset.append(store)
            else:
                action = uncer_action
            
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        
        self.mixing += mixing_decay
        if self.mixing > 1.0: self.mixing = 1.0
        
        return state, reward, done
        


if __name__ == "__main__":
    pass
    