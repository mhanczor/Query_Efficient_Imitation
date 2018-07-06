import numpy as np
from dagger import DAgger
# from DeepQ import DQNetwork
import random


"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
"""

class Efficient_DAgger(DAgger):
    """
    Only get expert actions when the learner is uncertain about the current state
    Reduces the amount of times the expert is queried
    
    Could still use the learner in the uncertain areas but see what the expert would have done?
    """
    def __init__(self, env, learner, expert, mixing=0.0, certainty_thresh=0.1):
        super(Efficient_DAgger, self).__init__(env, learner, expert, mixing)
        self.thresh = certainty_thresh
        # self.learner_predict = self.learner.uncertainAction
        self.learner_predict = self.learner.QBCAction
        
    def generateExpertSamples(self, mixing_decay=0.1):
        """
        General method for reduced sample DAgger
        Methods return the action that the learner should take, and some arbitrary confidence value
        """
        
        # Initialize environment and run a mixed trajectory
        # import pdb; pdb.set_trace()
        total_reward = 0
        state = self.env.reset()
        done = False
        expert_samples = 0
        while not done:
            
            learner_action, uncertainty_val = self.learner_predict(state)
            # print(action_var)
            # import pdb; pdb.set_trace()  
            mixing_prob = random.random()
            if uncertainty_val >= self.thresh or mixing_prob >= self.mixing: 
                if mixing_prob >= self.mixing:
                    action = self.expert.selectAction(state)
                    expert_action = action
                else:
                    action = learner_action
                    expert_action = self.expert.selectAction(state)
                store = state.tolist()
                store.append(expert_action)
                self.dataset.append(store)
                expert_samples += 1
            else:
                action = learner_action
            
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        
        self.mixing += mixing_decay
        if self.mixing > 1.0: self.mixing = 1.0
                
        return expert_samples, _
        
class Entropy_DAgger(Efficient_DAgger):
    """
    This approach samples from the policy output to select actions and 
    uses entropy to determine when to sample from the expert
    """
    
    def __init__(self, env, learner, expert, mixing=0.0, certainty_thresh=0.1):
        super(Entropy_DAgger, self).__init__(env, learner, expert, mixing, certainty_thresh)
        self.thresh = certainty_thresh
        self.learner_predict = self.entropyAction
        
    def entropyAction(self, state):
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
        # print entropy
        
        return action, entropy

class Random_DAgger(Efficient_DAgger):
    """
    A baseline to randomly select when to sample the expert
    The threshold sets the probability that an expert samples an action
    """
    
    def __init__(self, env, learner, expert, mixing=0.0, certainty_thresh=0.1):
        super(Random_DAgger, self).__init__(env, learner, expert, mixing, certainty_thresh)
        self.thresh = (1-certainty_thresh)
        self.learner_predict = self.selectAction
        
    def selectAction(self, state):
        action = self.learner.selectAction(state)
        rand_val = random.random()

        return action, rand_val

if __name__ == "__main__":
    pass
    