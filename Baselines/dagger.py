import numpy as np
import tensorflow as tf

# from DeepQ import DQNetwork
import random


"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
"""


class DAgger(object):
    
    def __init__(self, env, learner, expert, mixing=0.0):
        self.env = env
        self.expert = expert
        self.learner = learner
        self.mixing = mixing
        
        self.dataset = []
    
    def updateAgent(self, epochs=10, batch_size=32):
        if len(self.dataset) == 0:
            import pdb; pdb.set_trace()
            print("WARNING: No data available to train")
            return 0
        
        for ep in range(epochs):
            random.shuffle(self.dataset)
            total_loss = 0.
            i = 0
            while i < len(self.dataset):
                # import pdb; pdb.set_trace()
                batch = self.dataset[i:i+batch_size]
                loss = self.learner.update(np.array(batch))
                total_loss += loss
                i += batch_size
            average_loss = total_loss / len(self.dataset)
            print("\t Epoch {} loss {}".format(ep, average_loss))
        return average_loss
    
    def runEpisode(self, agent, render=False):
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            action = agent.selectAction(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if render:
                self.env.render()
        
        return total_reward
    
    def generateExpertSamples(self, mixing_decay=0.1):
        
        # Initialize environment and run a mixed trajectory
        total_reward = 0
        state = self.env.reset()
        done = False
        expert_samples = 0
        while not done:
            # Mix policies by randomly choosing between them
            if random.random() >= self.mixing:
                action = self.expert.selectAction(state)
                expert_action = action
            else:
                action = self.learner.selectAction(state)
                expert_action = self.expert.selectAction(state)
            #Aggregate expert data
            store = state.tolist()
            store.append(expert_action)
            self.dataset.append(store)
            state, reward, done, _ = self.env.step(action)
            
            expert_samples += 1
            total_reward += reward
        
        self.mixing += mixing_decay
        if self.mixing > 1.0: self.mixing = 1.0
        
        return state, reward, done, expert_samples
        
    def trainAgent(self, episodes=100, mixing_decay=0.1):
        
        validation = []
        
        # Run an initial validation to get starting agent reward
        valid_reward = 0
        for i in range(100):
            valid_reward += self.runEpisode(self.learner)
        validation.append(valid_reward/100.)
        
        stats = []
        for ep in range(episodes):
            _, _, _, expert_samples = self.generateExpertSamples(mixing_decay=mixing_decay)
            self.updateAgent()
            
            valid_reward = 0
            for i in range(100):
                valid_reward += self.runEpisode(self.learner)/100.0
            validation.append(valid_reward)
            print("Episode: {} reward: {} expert_samples: {}".format(ep, valid_reward, expert_samples))
            stats.append([ep, valid_reward, expert_samples])
        
        return validation, stats


if __name__ == "__main__":
    pass
    