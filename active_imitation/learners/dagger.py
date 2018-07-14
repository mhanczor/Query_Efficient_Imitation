import numpy as np
import tensorflow as tf
import os
import random

"""
Running things to fix:
Be able to revert to last best weights if validation is lower than previous?
    Don't actually need to do this, in practice just assume that the last trained policy is the best
"""

class DAgger(object):
    
    def __init__(self, env, learner, expert, agg_buffer, mixing=0.0, continuous=False):
        self.env = env
        self.expert = expert
        self.learner = learner
        self.mixing = mixing
        self.continuous = continuous
        
        self.dataset = agg_buffer
        
        self._logParameters()        
    
    def _logParameters(self):
    #     """
    #     Save all the parameters to a file and print out to the screen to start
    #     """
        env_name = str(self.env.unwrapped)
        learner = str(self.learner)
        mode = str(self)
        mixing = self.mixing
        dropout = self.learner.dropout_rate
        filepath = self.learner.filepath
        
        print("\n \n *** Now training a imitation network with the following parameters: *** \n \
        Environment Name: {} \n \
        Learner Type: {} \n \
        Sample Selection: {} \n \
        Continuous?: {} \n \
        Initial Mixing: {} \n \
        Initial Dropout: {} \n \
        \n \
        File Location: {} \n ".format(env_name, learner, mode, 
        self.continuous, mixing, dropout, filepath))    
    
    def updateAgent(self, epochs=10, batch_size=32):
        """
        Perform batch updating of the learner network
        """
                
        samples = self.dataset.size
        if samples == 0:
            # import pdb; pdb.set_trace()
            print("WARNING: No data available to train")
            return 0
            
        indices = list(range(samples))
        for ep in range(epochs):
            random.shuffle(indices)
            total_loss = 0.
            i = 0
            while i < samples:
                batch = self.dataset.sample(indices[i:i+batch_size])
                loss = self.learner.update(batch)
                total_loss += loss
                i += batch_size
            #TODO make sure this average math works out, what loss does TF return with batches?
            average_loss = total_loss / samples
            # print("\t Epoch {} loss {}".format(ep, average_loss))
        return average_loss
    
    def runEpisode(self, agent, render=False):
        # TODO will need a continuous action version of validation? 
        total_reward = 0
        state = self.env.reset()
        done = False
        episode_length = 0
        correct_labels = 0
        success = 0.0
        while not done:
            action = agent.sampleAction(state).squeeze() # sampleAction returns 2d arrays, need 1D
            expert_action = self.expert.sampleAction(state)
            if not self.continuous: # Can only compare actions in discrete action space
                if action == expert_action:
                    correct_labels += 1
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            episode_length += 1
            if self.continuous: # If using a robot env, check for success
                success = info['is_success']
            if render:
                self.env.render()
        
        return total_reward, correct_labels, episode_length, success
    
    def generateExpertSamples(self, mixing_decay=0.1):
        # Initialize environment and run a mixed trajectory
        total_reward = 0
        state = self.env.reset()
        done = False
        expert_samples = 0
        # import ipdb; ipdb.set_trace()
        while not done:
            # Mix policies by randomly choosing between them
            if random.random() >= self.mixing:
                action = self.expert.sampleAction(state)
                expert_action = action
            else:
                action = self.learner.sampleAction(state).squeeze()
                expert_action = self.expert.sampleAction(state)
            #Aggregate expert data
            self.dataset.store(state, expert_action)
            state, reward, done, _ = self.env.step(action)
            
            expert_samples += 1
            total_reward += reward
        
        self.mixing += mixing_decay
        if self.mixing > 1.0: self.mixing = 1.0
        
        return expert_samples, _
        
    def validateAgent(self, valid_runs):
        valid_reward = 0
        total_correct_labels, total_steps, total_success = 0, 0, 0.
        for i in range(valid_runs):
            cur_reward, correct_labels, ep_length, success = self.runEpisode(self.learner)
            valid_reward += cur_reward/valid_runs
            total_correct_labels += correct_labels
            total_steps += ep_length
            if success:
                total_success += 1
        avg_success = total_success / float(valid_runs)
        valid_acc = float(total_correct_labels) / total_steps 
        
        return valid_reward, valid_acc, avg_success
    
    def train(self, episodes=100, mixing_decay=0.1, train_epochs=10, 
                    save_images=False, image_filepath='./', save_rate=None):
        
        total_expert_samples = 0
        prev_samples = 0
        # Run an initial validation to get starting agent reward
        validation = []
        valid_runs = 5
        valid_reward, valid_acc, avg_successes = self.validateAgent(valid_runs)
        validation.append(valid_reward)
        
        reward_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Reward_per_Expert_Samples', simple_value=valid_reward)])
        self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
        if not self.continuous:
            accuracy_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_per_Expert_Samples', simple_value=valid_acc)])
            self.learner.writer.add_summary(accuracy_per_samples, global_step=total_expert_samples)
            variable_stat = valid_acc
        else:
            successes_per_sample  = tf.Summary(value=[tf.Summary.Value(tag='Success_per_Expert_Samples', simple_value=avg_successes)])
            self.learner.writer.add_summary(successes_per_sample, global_step=total_expert_samples)
            variable_stat = avg_successes
            
        stats = [[0, 0, 0, valid_reward, variable_stat, 0.]]
        print("Episode: {} reward: {} expert_samples: {}".format(0, valid_reward, 0))
        
        for ep in range(episodes):
            if save_images:
                # Save an image of the environment at every expert query
                if not os.path.isdir(image_filepath):
                    os.mkdir(image_filepath)
                from scipy.misc import imsave
                expert_samples, images, utility_measure = self.generateExpertSamples(mixing_decay=mixing_decay, save_image=True)
                for num, image in enumerate(images):
                    filename = image_filepath + 'episode_' + str(ep) + '_' + str(num) + '.png'
                    imsave(filename, image)
            else:
                expert_samples, _, utility_measure = self.generateExpertSamples(mixing_decay=mixing_decay)

            final_loss = self.updateAgent(epochs=train_epochs)
            total_expert_samples += expert_samples
            
            valid_reward, valid_acc, avg_successes = self.validateAgent(valid_runs)
            validation.append(valid_reward)
            
            # Is there a good way to clean this up?
            reward_per_samples = tf.Summary(value=[tf.Summary.Value(tag='Reward_per_Expert_Samples', simple_value=valid_reward)])
            self.learner.writer.add_summary(reward_per_samples, global_step=total_expert_samples)
            samples_per_episode = tf.Summary(value=[tf.Summary.Value(tag='Expert_Samples_per_Episode', simple_value=expert_samples)])
            self.learner.writer.add_summary(samples_per_episode, global_step=ep)
            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Final_Training_Loss_per_Episode', simple_value=final_loss)])
            self.learner.writer.add_summary(loss_summary, global_step=ep)
            
            if not self.continuous:
                accuracy_per_samples  = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_per_Expert_Samples', simple_value=valid_acc)])
                self.learner.writer.add_summary(accuracy_per_samples, global_step=total_expert_samples)
                variable_stat = valid_acc
            else:
                successes_per_sample  = tf.Summary(value=[tf.Summary.Value(tag='Success_per_Expert_Samples', simple_value=avg_successes)])
                self.learner.writer.add_summary(successes_per_sample, global_step=total_expert_samples)
                variable_stat = avg_successes
            
            utility_summary = tf.Summary(value=[tf.Summary.Value(tag='Sample_Utility', simple_value=utility_measure)])
            self.learner.writer.add_summary(utility_summary, global_step=total_expert_samples)
            
            print("Episode: {} reward: {} expert_samples: {}".format(ep+1, valid_reward, expert_samples))
            stats.append([ep+1, total_expert_samples, expert_samples, valid_reward, variable_stat, utility_measure])
            # if # we have crossed the threshold of expert samples
            
        valid_reward, valid_acc, avg_successes = self.validateAgent(5)
        validation.append(valid_reward)
        print("\n Training Complete")
        print("Final validation reward: {} total expert samples: {}".format(valid_reward, total_expert_samples))
        
        
        return validation, stats


if __name__ == "__main__":
    pass
    