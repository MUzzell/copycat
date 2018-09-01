
import numpy as np
import tensorflow as tf
import argparse as ap
import gym

import pdb

from dqn import agent as dqn_a

parser = ap.ArgumentParser("Test DQN Atari runner")
parser.add_argument("game", default="space_invaders")

parser.add_argument("-f", "--frame_limit", default=50000)
parser.add_argument("-l", "--learn_start", default=5000)
parser.add_argument("-e", "--eval_freq", default=10000)
parser.add_argument("-el", "--eval_len", default=5000)
parser.add_argument("-rm", "--replay_memory", default=30000)

args = parser.parse_args()

pdb.set_trace()

env = gym.make(args.game)

agent = dqn_a.NeuralQLearner(
    env.action_space.n,
    (1, 0.1, args.replay_memory, args.learn_start),
    (0.00025, 0.00025, args.replay_memory, args.learn_start),
    discount=0.99, state_dim=7056, replay_memory=args.replay_memory)

step = 0
