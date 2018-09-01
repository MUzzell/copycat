
from .e_greedy import AEGreedy
from .transition_table import TransitionTable

import numpy as np
import tensorflow as tf


class NeuralQLearner(object):

    def __init__(self, actions, greedy_conf, l_greedy_conf, *args, **kwargs):

        self.actions = actions

        self.greedy = AEGreedy(*greedy_conf)
        self.l_greedy = AEGreedy(*l_greedy_conf)

        self.network = None
        self.target_net = None

        self.discount = kwargs.get('discount', 0.99)
        self.state_dim = kwargs.get('state_dim', 7056)
        self.hist_len = kwargs.get('hist_len', 4)
        self.hist_spacing = kwargs.get('hist_spacing', 1)
        self.hist_type = kwargs.get('hist_type', 'linear')
        self.replay_memory = kwargs.get('replay_memory', 30000)
        self.non_term_prob = kwargs.get('non_term_prob', )
        self.minibatch_size = kwargs.get('minibatch_size', 32)
        self.buffer_size = kwargs.get('buffer_size', 512)
        self.valid_size = kwargs.get('valid_size', 500)
        self.target_q = kwargs.get('target_q', 10000)

        kwargs['n_actions'] = self.actions
        kwargs['state_dim'] = self.state_dim
        kwargs['hist_len'] = self.hist_len

        self.transitions = TransitionTable(*args, **kwargs)

    def step(self, state):

        ep = self.greedy.greedy(self.n_actions, self.steps)
        if np.random.uniform() < ep:
            return np.random.randint(self.n_actions)

        if len(state.shape) == 2:
            raise ValueError("State must be 3D!")

        q = self.run_net(state)

    def get_q_update(self, states, actions, rewards, s2, terms, update_qmax=True):

        if self.target_q:
            target_q_net = self.target_net
        else:
            target_q_net = self.network

        term = np.mul(term, -1) + 1

        raise NotImplemented("get_q_update run q2_max")
        q2_max = np.random.random(self.n_actions)

        q2 = np.mul(q2_max.copy() * self.discount, term)

        delta = r.copy()

        if self.rescale_r:
            delta /= self.r_max

        delta += q2

        raise NotImplemented("get_q_update run q_all")
        q_all = np.random.random(self.n_actions)
        q = np.array(q_all.shape[0])

        for i in range(q_all.shpae[0]):
            q[i] = q_all[i, a[i]]

        delta += q * -1

        if self.clip_delta:
            delta[delta >= self.clip_delta] = self.clip_delta
            delta[delta <= self.clip_delta] = -self.clip_delta

        targets = np.zeros((self.minibatch_size, self.n_actions))

        for i in range(np.min([self.minibatch_size, a.shape[0]])):
            targets[i, a[i]] = delta[i]

        return targets, delta, q2_max


    def qLearnMinibatch():

        assert self.transitions.size() > self.minibatch_size

        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        targets, delta, q2_max = self.get_q_update(s, a, r, s2, term, True)

        # TODO: ZERO GRADIENTS
        raise NotImplemented("qLearnMinibatch zero gradients")

        raise NotImplemented("qLearnMinibatch backprop")

        raise NotImplemented("qLearnMinibatch add weight cost to grad")

        lr = self.l_greedy.greedy(self.steps)
        lr = np.max([lr, self.greedy.end])

        raise NotImplemented("qLearnMinibatch apply gradients")

        raise NotImplemented("qLearnMinibatch accumulate update")


    def perceive(session, reward, state, term, test=False):

        state = self.preprocess(state).float()

        if self.max_reward:
            reward = np.min([reward, self.max_reward])

        if self.min_reward:
            reward = np.max([reward, self.min_reward])

        if self.rescale_r:
            self.r_max = np.max([self.r_max, reward])

        self.transitions.add_recent(state, terminal)

        currentFullState = self.transitions.get_recent()

        if self.lastState and not test:
            self.transitions.add(self.lastState, self.lastAction, reward,
                                 self.lastTerm)

        if self.numSteps == self.l_start + 1 and not test:
            self.sample_validation_data()

        currState = self.transitions.get_recent()
        currState = currState.resize(1, unpack(self.input_dims))

        action_idx = 1
        if not term:
            action_idx = self.step(state)

        self.transitions.add_recent_action(action_idx)

        if (self.steps > self.l_start and not test
                and self.steps % self.update_freq == 0):
            for i in range(self.n_replay):
                self.qLearnMinibatch()

        if not test:
            self.steps = self.steps + 1

        self.last_state = state
        self.last_action = action_idx
        self.last_term = term

        if self.target_q and self.num_steps % self.target_q == 1:
            raise NotImplemented("perceive clone target net")

        if not term:
            return action_idx

        return 0

