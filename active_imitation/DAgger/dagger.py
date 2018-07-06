import numpy as np
import tensorflow as tf
import os

# from DeepQ import DQNetwork
import random


"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
    Don't actually need to do this, in practice just assume that the last trained policy is the best
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
            # print("\t Epoch {} loss {}".format(ep, average_loss))
        return average_loss
    
    def runEpisode(self, agent, render=False):
        total_reward = 0
        state = self.env.reset()
        done = False
        episode_length = 0
        correct_labels = 0
        while not done:
            action = agent.selectAction(state)
            expert_action = self.expert.selectAction(state)
            if action == expert_action:
                correct_labels += 1
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            episode_length += 1
            if render:
                self.env.render()
        
        return total_reward, correct_labels, episode_length
    
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
        
        return expert_samples, _
        
    def validateAgent(self, valid_runs):
        valid_reward = 0
        total_correct_labels, total_steps = 0, 0
        for i in range(valid_runs):
            cur_reward, correct_labels, ep_length = self.runEpisode(self.learner)
            valid_reward += cur_reward/valid_runs
            total_correct_labels += correct_labels
            total_steps += ep_length
        valid_acc = float(total_correct_labels) / total_steps 
        
        return valid_reward, valid_acc
    
    def trainAgent(self, episodes=100, mixing_decay=0.1, train_epochs=10, 
                    save_images=False, image_filepath='./'):
        # import ipdb; ipdb.set_trace()
        validation = []
        total_expert_samples = 0
        # Run an initial validation to get starting agent reward
        valid_runs = 10
        # import ipdb; ipdb.set_trace()
        valid_reward, valid_acc = self.validateAgent(valid_runs)
        
        validation.append(valid_reward)
        reward_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Reward_per_Expert_Samples', simple_value=valid_reward)])
        self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
        accuracy_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_per_Expert_Samples', simple_value=valid_acc)])
        self.learner.writer.add_summary(accuracy_per_samples, global_step=total_expert_samples)
        
        stats = []
        for ep in range(episodes):
            if save_images:
                if not os.path.isdir(image_filepath):
                    os.mkdir(image_filepath)
                from scipy.misc import imsave
                expert_samples, images = self.generateExpertSamples(mixing_decay=mixing_decay, save_image=True)
                for num, image in enumerate(images):
                    filename = image_filepath + 'episode_' + str(ep) + '_' + str(num) + '.png'
                    imsave(filename, image)
            else:
                expert_samples, _ = self.generateExpertSamples(mixing_decay=mixing_decay)
            # import ipdb; ipdb.set_trace()
            final_loss = self.updateAgent(epochs=train_epochs)
            total_expert_samples += expert_samples
            
            valid_reward, valid_acc = self.validateAgent(valid_runs)
            validation.append(valid_reward)
            
            reward_per_samples = tf.Summary(value=[tf.Summary.Value(tag='Reward_per_Expert_Samples', simple_value=valid_reward)])
            self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
            samples_per_episode = tf.Summary(value=[tf.Summary.Value(tag='Expert_Samples_per_Episode', simple_value=expert_samples)])
            self.learner.writer.add_summary(samples_per_episode, global_step=ep)
            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Final_Training_Loss_per_Episode', simple_value=final_loss)])
            self.learner.writer.add_summary(loss_summary, global_step=ep)
            accuracy_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_per_Expert_Samples', simple_value=valid_acc)])
            self.learner.writer.add_summary(accuracy_per_samples, global_step=total_expert_samples)
            
            print("Episode: {} reward: {} expert_samples: {}".format(ep, valid_reward, expert_samples))
            stats.append([ep, valid_reward, expert_samples])
        
        return validation, stats


if __name__ == "__main__":
    pass
    