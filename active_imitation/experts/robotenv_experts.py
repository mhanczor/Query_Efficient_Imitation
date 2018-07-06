import sys, argparse
import numpy as np
import pickle


class RoboticEnv_Expert(object):
    def __init__(self, policy_file, request_q=False):
        self.request_q = request_q
        with open(policy_file, 'rb') as f:
            self.policy = pickle.load(f)
        pass
    
    def selectAction(self, state):
        """ 
        Action selecton for fetch tasks
        state - a dictionary of [observations, achieved_goal, desired_goal]
        """
        
        o = state['observation']
        ag = state['achieved_goal']
        g = state['desired_goal']
        
        policy_output = self.policy.get_actions(o, ag, g,
                        noise_eps=0.,
                        random_eps=0.)
        if self.request_q:
            action, q_val = policy_output
        else:
            action = policy_output
        
        return action
        

if __name__ == '__main__':
    # Run a series of actions to test the expert
    import gym, gym.spaces
    policy_file = '/tmp/openai-2018-07-03-17-00-07-524061/policy_best.pkl'
    env = gym.make('FetchReach-v1')
    expert = FetchReach_Expert(policy_file)
    state = env.reset()
    for i in range(10):
        action = expert.selectAction(state)
        print('Step {}'.format(i))
        print(action)
        state = env.step(action)
        success = state[1]
        state = state[0]
        print(state)
        