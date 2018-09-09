import argparse

parser = argparse.ArgumentParser(description='A3C')

# Game Environment
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='game environment (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--repeat-action-steps', type=int, default=4,
                    help='Repeat each action for a few steps (default: 4)')
parser.add_argument('--colour-frame', action='store_true', default=False,
                    help='If true, use colour rather than monochrome image')

# Batch training
parser.add_argument('--num-actors', type=int, default=3,
                    help='batchsize of training experience')
