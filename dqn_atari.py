
import numpy as np
import argparse as ap
import retro
import time

import pdb

from dqn import agent as dqn_a


def run_step(env, action_idx, frameskip=4):

    state = None
    reward = 0.0
    term = False

    for i in range(frameskip):
        if term:
            break

        state, sreward, term, _ = env.step(action_idx)
        reward += sreward

    return state, reward, term


def run_evaluation(env, agent, eval_len):
    screen, reward, term, = (env.reset(), 0, False)

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0

    eval_start = time.time()

    for i in range(args.eval_len):
        action_idx = agent.perceive(reward, screen, term, test=True)

        screen, reward, term = run_step(env, action_idx, 4)

        episode_reward += reward
        if reward != 0:
            nrewards += 1

        if term:
            total_reward += episode_reward
            episode_reward = 0
            nepisodes += 1
            screen, reward, term = (env.reset(), 0, False)

    eval_time = time.time() - eval_start
    print("TODO: compute_validation_statistics")

    total_reward /= np.max([1, nepisodes])

    print("TODO: Save best network")
    print("TODO: V, TD error & qMax histories")

    print(
        "Steps: %d (frames %d)\n"
        "Reward: %.2f Episodes: %d"
        "Eval time: %d Eval rate: %.2f".format(
            args.eval_len, args.eval_len*4,
            total_reward, nepisodes,
            eval_time.total_seconds(),
            args.eval_len*4 / eval_time.total_seconds()
        )
    )


parser = ap.ArgumentParser("Test DQN Atari runner")
parser.add_argument("game", default="SpaceInvaders-Atari2600",
                    choices=retro.data.list_games())

parser.add_argument("-f", "--frame_limit", default=50000)
parser.add_argument("-l", "--learn_start", default=5000)
parser.add_argument("-e", "--eval_freq", default=10000)
parser.add_argument("-el", "--eval_len", default=5000)
parser.add_argument("-rm", "--replay_memory", default=30000)

args = parser.parse_args()

env = retro.make(args.game)

agent = dqn_a.NeuralQLearner(
    env.action_space.n,
    (1, 0.1, args.replay_memory, args.learn_start),
    (0.00025, 0.00025, args.replay_memory, args.learn_start),
    learn_start=args.learn_start, discount=0.99,
    state_dim=7056, replay_memory=args.replay_memory)

screen, reward, term = (env.reset(), 0, False)

input_state = np.zeros((4, 84, 84), dtype=np.float32)

step = 0

while step < args.frame_limit:

    step += 1
    action_idx = agent.perceive(reward, screen, term)

    if not term:
        screen, reward, term = run_step(env, action_idx, 4)
    else:
        screen, reward, term = (env.reset(), 0, False)

    if step % args.eval_freq == 0 and step > args.learn_start:
        run_evaluation(env, agent, args.eval_len)
