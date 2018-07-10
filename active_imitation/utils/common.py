import argparse


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', dest='num_episodes', type=int, default=10, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help="The learning rate.")
    parser.add_argument('--env', dest='env_name', type=str, default='CartPole-v1',help="Environment Name")
    parser.add_argument('--file', dest='file_name', type=str, default='-1', help="Filename to save.")
                              
    parser.add_argument('--load', dest='load_model', action='store_true', help="Whether to load the model.")
    parser.set_defaults(load=False)
    
    parser.add_argument('--thresh', dest='uncertainty_threshold', type=float, default=0.1, help="Threshold for efficient dagger.")

    return parser.parse_args()
