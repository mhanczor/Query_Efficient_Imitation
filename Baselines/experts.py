import numpy as np
import tensorflow as tf

import gym


""" Expert Classes """

class CartPole_iLQR(object):
    """
    Credit to Wesley Yue
    https://gist.github.com/WesleyYue/27352f8cf5a835d05b558e8fdda9b59d
    """
    
    def __init__(self, env):
        self.q_avail = False
        self.env = env
    
    def selectAction(self, x):
        
        # x, xdot, theta, thetadot

        gamma = (4.0 / 3.0 - self.env.masspole / self.env.total_mass)

        a = -self.env.gravity * self.env.masspole / (self.env.total_mass * gamma)
        b = (1.0 / self.env.total_mass * (1 + self.env.masspole / (self.env.total_mass * gamma)))
        c = self.env.gravity / (self.env.length * gamma)
        d = -1.0 / (self.env.total_mass * self.env.length * gamma)

        tau = self.env.tau

        F = np.array([
            [1, tau,       0,   0,       0],
            [0,   1, tau * a,   0, tau * b],
            [0,   0,       1, tau,       0],
            [0,   0, tau * c,   1, tau * d],
          ])

        C = np.array([
            [1,  0, 0,  0,   0],
            [0,  0, 0,  0,   0],
            [0,  0, 1,  0,   0],
            [0,  0, 0,  0,   0],
            [0,  0, 0,  0,   1],
          ])

        c = np.array([0, 0, 0, 0, 0]).T

        frame = 0
        i = 0
        while i < 1: # changed from while 1
            i += 1
            Ks = []
            T = 100
            # V = np.zeros((4, 4))
            # v = np.zeros((4))
            V = C[:4, :4]
            v = np.zeros((4))
            for t in range(T, -1, -1):
                # Qt
                Qt = C + np.matmul(F.T, np.matmul(V, F))
                qt = c + np.matmul(F.T, v)


                Quu = Qt[-1:,-1:]
                Qux = Qt[-1:,:-1]
                Qxu = Qt[:-1, -1:]

                qu = qt[-1:]

                Qut_inv = np.linalg.inv(Quu)

                Kt = -np.matmul(Qut_inv, Qux)
                kt = -np.matmul(Qut_inv, qu)

                Ks.append((Kt, kt))

                V = Qt[:4, :4] + np.matmul(Qxu, Kt) + np.matmul(Kt.T, Qux) + np.matmul(Kt.T, np.matmul(Quu, Kt))
                v = qt[:4] + np.matmul(Qxu, kt) + np.matmul(Kt.T, qu) + np.matmul(Kt.T, np.matmul(Quu, kt))

                Kt, kt = Ks[-1]
                ut = np.matmul(Kt, x.reshape((1, -1)).T) + kt

            if ut > 0.0:
              ut = self.env.force_mag
              action = 1
            else:
              ut = -self.env.force_mag
              action = 0


            # xu = np.hstack([x, ut])
            # my_guess = np.matmul(F, xu.T)
            # x, reward, done, info = env.step(action)
        return action

        

class CartPole_SubExpert(object):

    def __init__(self):
        
        self.q_avail = True # This expert can provide a Q-val estimate
        
        #Starting a new session for the expert
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        # Set expert model and weights here
        filepath = 'saved_weights/CartPoleV1_209/checkpoints/'
        
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(filepath) + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(filepath))
            self.x = self.graph.get_tensor_by_name('Input/Features:0')
            self.y = self.graph.get_tensor_by_name('Output/Q_predicted:0') 
               
        
    def _predict(self, state):
        state = np.atleast_2d(state)
        feed_dict = {self.x:state}
        return self.sess.run(self.y, feed_dict)
    
    def selectAction(self, state):
        # Returns an expert action
        q_vals = self._predict(state)
        action = np.argmax(q_vals)
        return action
    
    def predictValue(self, state, action):
        pass
        
        

class LunarLander_Expert(object):
    def __init__(self, continuous=False):
        self.continuous = continuous
    
    def selectAction(self, s):
        # Heuristic for:
        # 1. Testing. 
        # 2. Demonstration rollout.
        angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4: angle_targ = -0.4
        hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]: # legs have contact
            angle_todo = 0
            hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

        if self.continuous:
            a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
            elif angle_todo < -0.05: a = 3
            elif angle_todo > +0.05: a = 1
        return a


if __name__ == "__main__":
    
    env = gym.make('CartPole-v1')
    
    state = env.reset()
    state = np.atleast_2d(state)
    expert = CartPole_Expert()
    
    print(expert.predictAction(state))
    