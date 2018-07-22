import tensorflow as tf
from baselines.ppo2.policies import CnnPolicy
from baselines.ppo2.ppo2 import Model
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env
import time
import gym
import os
from active_imitation.experts import trained_models


class SpaceInvadersExpert(object):
    def __init__(self, env_dims):
        prefix = os.path.dirname(trained_models.__file__)
        exp_filepath = os.path.join(prefix, 'SpaceInvadersNoFrameskip-v0/78800_atari_model.ckpt')
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        obs_space = env_dims['observation']
        act_space = env_dims['action']
        
        self.model = Model(policy=CnnPolicy, ob_space=obs_space, ac_space=act_space, nbatch_act=1, 
                    nbatch_train=1, nsteps=1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
                    
        saver = tf.train.Saver()
        saver.restore(self.sess, exp_filepath)
        
    def sampleAction(self, obs):
        action, value, state, neglogpac = self.model.step(obs)
        return action
    
    def close(self):
        tf.reset_default_graph()
        self.sess.close()


if __name__ == "__main__":
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind
    import numpy as np
    env_id = 'SpaceInvadersNoFrameskip-v0'
    # The make atarri and vecframestack take a bit of the work out of reducing the
    # environment down from a 260px image and also i believe handle the color channel stuff, so worthwhile to keep
    # 
    wrapper_kwargs = {'episode_life':False}
    env = VecFrameStack(make_atari_env(env_id, 1, 0, wrapper_kwargs=wrapper_kwargs), 4)
    # env = VecFrameStack(wrap_deepmind(make_atari(env_id), episode_life=False), 4)
    env_dims = {'observation':env.observation_space, 'action':env.action_space}
    
    obs = env.reset()
    env.render()
    time.sleep(0.05)
    expert = SpaceInvadersExpert(env_dims)
    for i in range(1):
        done = False
        j = 1
        total_reward = 0
        while not done:
            action = expert.sampleAction(obs)
            env.render()
            # action = [env.action_space.sample()]
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(reward)
            time.sleep(0.01)
        
#######
"""
If you want an easy way to render at a large size to take a video from:
use:

env = wrap_deepmind(make_atari(env_id), frame_stack=True)
obs = env.reset()
obs = np.array(obs)
env.render()

The observation is a LazyArray and needs to be converted to an array by using np.array()


    env = wrap_deepmind(make_atari(env_id), frame_stack=True)
    obs = env.reset()
    obs = np.array(obs)
    env.render()
    time.sleep(0.05)
    done = False
    import ipdb; ipdb.set_trace()
    i = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        obs = np.array(obs)
        i += 1
        env.render()
        if i > 50:
            import ipdb; ipdb.set_trace()
            print(info, done)
"""
        
