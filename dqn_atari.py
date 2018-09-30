
import numpy as np
import tensorflow as tf
import argparse as ap
import gym
from PIL import Image

import pdb

from dqn import agent as dqn_a
from dqn.img_ops import preproc


def run_step(env, action_idx, frameskip=4, state_dim=(210,160,3)):

    state = np.zeros((frameskip,) + state_dim)
    reward = 0.0
    term = False

    for i in range(frameskip):
        if term:
            state[i, :] = state[i-1, :]
        else:
            state[i, :], reward, term, _ = env.step(action_idx)
            reward += reward

    assert state.shape == (frameskip,) + state_dim

    return state, reward, term


def reset_env(env, frameskip=4):

    screen = env.reset()
    return np.array([screen for i in range(frameskip)]), 0, False


parser = ap.ArgumentParser("Test DQN Atari runner")
parser.add_argument("game", default="space_invaders")

parser.add_argument("-f", "--frame_limit", default=50000)
parser.add_argument("-l", "--learn_start", default=5000)
parser.add_argument("-e", "--eval_freq", default=10000)
parser.add_argument("-el", "--eval_len", default=5000)
parser.add_argument("-rm", "--replay_memory", default=30000)

args = parser.parse_args()

env = gym.make(args.game)

agent = dqn_a.NeuralQLearner(
    env.action_space.n,
    (1, 0.1, args.replay_memory, args.learn_start),
    (0.00025, 0.00025, args.replay_memory, args.learn_start),
    learn_start=args.learn_start, discount=0.99,
    state_dim=7056, replay_memory=args.replay_memory)

screen = env.reset()
reward = 0
term = False

input_state = np.zeros((4, 84, 84), dtype=np.float32)

step = 0

while step < args.frame_limit:

    step += 1
    pdb.set_trace()
    action_idx = agent.perceive(reward, screen, term)

    if not term:
        screen, reward, term, _ = env.step(action_idx)
    else:
        screen = env.reset()
        reward = 0
        term = False


for i in range(4):
    img = preproc((env.step(0))[0], (84,84))
    input_state[i,:] = img

action = agent.target_net.predict(np.expand_dims(input_state, axis=0))
