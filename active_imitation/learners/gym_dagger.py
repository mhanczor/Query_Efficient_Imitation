import sys, argparse
import numpy as np
import tensorflow as tf
import gym
from dagger import DAgger
from efficient_dagger import Efficient_DAgger, Entropy_DAgger, Random_DAgger
from hindsight_DAgger import Hindsight_DAgger
from active_imitation.experts import CartPole_SubExpert, CartPole_iLQR, LunarLander_Expert

from scipy import stats

import matplotlib

# Build a learner model

def plot(stats):
    pass
    
        
def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-config-path', dest='model_config_path',
    #                     type=str, default='LunarLander-v2-config.json',
    #                     help="Path to the model config file.")
    parser.add_argument('--episodes', dest='num_episodes', type=int,
                        default=10, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser_group = parser.add_mutually_exclusive_group(required=False)
    # parser_group.add_argument('--render', dest='render',
    #                           action='store_true',
    #                           help="Whether to render the environment.")
    # parser_group.add_argument('--no-render', dest='render',
    #                           action='store_false',
    #                           help="Whether to render the environment.")
    # parser.set_defaults(render=False)
    
    parser.add_argument('--env', dest='env_name',
                              type=str, default='CartPole-v1',
                              help="Environment Name")
    
    parser.add_argument('--file', dest='file_name',
                              type=str, default='-1',
                              help="Filename to save.")
                              
    parser.add_argument('--load', dest='load_model',
                              action='store_true',
                              help="Whether to load the model.")
    parser.set_defaults(load=False)
    
    parser.add_argument('--thresh', dest='uncertainty_threshold', type=float,
                        default=0.1, help="Threshold for efficient dagger.")
    
    parser.add_argument('--loadfile', dest='load_file', type=str,
                              default='',
                              help="Load model")
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    episodes = args.num_episodes
    filename = args.file_name
    env_name = args.env_name
    var_thresh = args.uncertainty_threshold
        
    filepath = 'tests/' + env_name + '/'+filename+'/'
    
    mixing = 1.0
    mixing_decay = 1.0
    train_epochs = 10
    dropout_rate = 0.01
    random_sample = False    
    
    env = gym.make(env_name)
    sess = tf.Session()
    agent = GymAgent(sess, env, lr=0.001, dropout_rate=dropout_rate, filepath=filepath)
    
    # expert = CartPole_SubExpert()
    if env_name[:-3] == 'CartPole':
        expert = CartPole_iLQR(env.env)
    elif env_name[:-3] == 'LunarLander':
        expert = LunarLander_Expert()
    else:
        raise ValueError
    
    # learner = DAgger(env, agent, expert, mixing=0.0)
    # learner = Efficient_DAgger(env, 
    #                           agent, 
    #                           expert, 
    #                           mixing=1.0, 
    #                           certainty_thresh=var_thresh)
    # learner = Entropy_DAgger(env, 
    #                         agent, 
    #                         expert, 
    #                         mixing=0.0, 
    #                         certainty_thresh=var_thresh)
    # learner = Random_DAgger(env,
    #                         agent,
    #                         expert,
    #                         mixing=1.0,
    #                         certainty_thresh=var_thresh)
    learner = Hindsight_DAgger(env,
                                agent,
                                expert,
                                mixing=mixing,
                                random_samp=random_sample)
    
    rewards, stats = learner.trainAgent(episodes=episodes, 
                                        mixing_decay=mixing_decay,
                                        train_epochs=train_epochs,
                                        save_images=False,
                                        image_filepath=filepath+'images/')
    
    with open(filepath + 'stats.csv', 'a') as f:
        for line in stats:
            line = map(str, line)
            f.write(', '.join(line) + '\n')
    
    for i in range(5):
        # learner.runEpisode(expert, render=True)
        learner.runEpisode(agent, render=True)
    

def sampleRun():
    episodes = 10
    filename = 'NA'
    env_name = 'LunarLander-v2'
    var_thresh = 0.5
        
    filepath = 'tmp/' + env_name + '/'+filename+'/'
    
    mixing = 1.0
    mixing_decay = 1.0
    train_epochs = 10
    dropout_rate = 0.01
    random_sample = False    
    
    env = gym.make(env_name)
    sess = tf.Session()
    agent = GymAgent(sess, env, lr=0.001, dropout_rate=dropout_rate, filepath=filepath)
    
    # expert = CartPole_SubExpert()
    if env_name[:-3] == 'CartPole':
        expert = CartPole_iLQR(env.env)
    elif env_name[:-3] == 'LunarLander':
        expert = LunarLander_Expert()
    else:
        raise ValueError
        
    learner = Hindsight_DAgger(env,
                                agent,
                                expert,
                                mixing=mixing,
                                random_sample=random_sample)
    
    rewards, stats = learner.train(episodes=episodes, 
                                    mixing_decay=mixing_decay,
                                    train_epochs=train_epochs,
                                    save_images=False,
                                    image_filepath=filepath+'images/')
    return rewards, stats


if __name__ == "__main__":
    main(sys.argv)
        
        