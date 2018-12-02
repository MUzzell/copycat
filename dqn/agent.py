
from .e_greedy import AEGreedy
from .transition_table import TransitionTable
from .img_ops import preproc
import numpy as np

import torch
from .models.torch_dqn_convnet import build_network_torch_class as build_network

import logging
logger = logging.getLogger(__name__)


def build_action_array(n_actions, action_idx):
    out = np.zeros(n_actions)
    out[action_idx] = 1
    return out


class NeuralQLearner(object):

    def __init__(self, n_actions, greedy_conf, l_greedy_conf, *args, **kwargs):

        self.n_actions = n_actions

        self.greedy = AEGreedy(*greedy_conf)
        self.l_greedy = AEGreedy(*l_greedy_conf)

        self.l_start = kwargs.get('learn_start', 50000)
        self.discount = kwargs.get('discount', 0.99)
        self.update_freq = kwargs.get('update_freq', 4)
        self.n_replay = kwargs.get('n_replay', 1)
        self.state_dim = kwargs.get('state_dim', 7056)
        self.hist_len = kwargs.get('hist_len', 4)
        self.ncols = kwargs.get('ncols', 1)
        self.hist_spacing = kwargs.get('hist_spacing', 1)
        self.hist_type = kwargs.get('hist_type', 'linear')
        self.replay_memory = kwargs.get('replay_memory', 30000)
        self.non_term_prob = kwargs.get('non_term_prob', )
        self.minibatch_size = kwargs.get('minibatch_size', 32)
        self.buffer_size = kwargs.get('buffer_size', 512)
        self.valid_size = kwargs.get('valid_size', 500)
        self.target_q = kwargs.get('target_q', 10000)
        self.clip_delta = kwargs.get('clip_delta', 1)
        self.wc = kwargs.get('weight_cost', 0)

        self.max_reward = kwargs.get('max_reward', 1)
        self.min_reward = kwargs.get('min_reward', -1)
        self.rescale_r = kwargs.get('rescale_r', True)

        self.input_dims = (self.hist_len * self.ncols, 84, 84)

        kwargs['n_actions'] = self.n_actions
        kwargs['state_dim'] = self.state_dim
        kwargs['hist_len'] = self.hist_len

        self.transitions = TransitionTable(*args, **kwargs)

        self.network = build_network(self.input_dims, self.n_actions)
        self.target_net = build_network(self.input_dims, self.n_actions)

        logger.info("Generated network: %r", self.network)

        self.q_max = 1
        self.r_max = 1
        self.num_steps = 0
        self.last_state = None
        self.last_action = None
        self.last_term = None

    def step(self, state):

        ep = self.greedy.greedy(self.num_steps)
        if np.random.uniform() < ep:
            return np.random.randint(self.n_actions)

        if len(state.shape) == 2:
            raise ValueError("State must be 3D!")
        q = self.network.forward(state)
        q = q[0].detach().numpy()
        maxq = q[0]
        besta = [0]

        for i in range(1, self.n_actions):
            if q[i] > maxq:
                besta = [i]
                maxq = [q[i]]
            elif q[i] == maxq:
                besta.append(i)

        self.bestq = maxq
        return np.random.choice(besta)

    def sample_validation_data(self):
        assert self.transitions.size > self.valid_size
        s, a, r, s2, term = self.transitions.sample(self.valid_size)
        s2 = s2.reshape(self.valid_size, self.hist_len, 84, 84)
        s = s.reshape(self.valid_size, self.hist_len, 84, 84)

        self.valid_s = np.copy(s)
        self.valid_a = np.copy(a)
        self.valid_r = np.copy(r)
        self.valid_s2 = np.copy(s2)
        self.valid_term = np.copy(term)

    def sample_validation_statistics(self):
        self.sample_validation_data()

        targets, delta, q2_max, _ = self.get_q_update(
            self.valid_s, self.valid_a, self.valid_r,
            self.valid_s2, self.valid_term
        )

        delta = delta.detach().numpy()
        q2_max = q2_max.detach().numpy()

        return (
            self.q_max * np.mean(q2_max),
            np.mean(np.abs(delta))
        )

    def get_q_update(
        self,
        states, actions, rewards, s2, term,
        update_qmax=True
    ):

        if self.target_q:
            target_q_net = self.target_net
        else:
            target_q_net = self.network

        term = torch.from_numpy((term * -1) + 1).float()

        q2_max = target_q_net.forward(s2).max(1)[0]
        # q2_max = np.random.random(self.n_actions)
        q2 = torch.mul(q2_max * self.discount, term)

        # delta is not touched before q+-1 line,
        # is this pointless?
        delta = torch.from_numpy(rewards.copy()).float()
        if self.rescale_r:
            delta = delta / self.r_max
        delta = torch.add(delta, q2)

        q_all = self.network.forward(states)

        # q_all = np.random.random(self.n_actions)
        q = torch.zeros(q_all.shape[0])

        for i in range(q_all.shape[0]):
            q[i] = q_all[i, int(actions[i])]

        delta.add(-1, q)

        if self.clip_delta:
            delta[delta >= self.clip_delta] = self.clip_delta
            delta[delta <= self.clip_delta] = -self.clip_delta

        targets = torch.zeros(self.minibatch_size, self.n_actions)

        for i in range(np.min([self.minibatch_size, actions.shape[0]])):
            targets[i, int(actions[i])] = delta[i]

        return targets, delta, q2_max, q_all

    def qLearnMinibatch(self):

        assert self.transitions.size > self.minibatch_size

        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        s2 = s2.reshape(self.minibatch_size, self.hist_len, 84, 84)
        s = s.reshape(self.minibatch_size, self.hist_len, 84, 84)
        targets, delta, q2_max, q_all = self.get_q_update(
            s, a, r, s2, term, True
        )

        self.network.backward(q_all, targets)

        t = np.max([0, self.num_steps - self.l_start])
        lr = (self.l_greedy.start - self.l_greedy.end) * (self.l_greedy.endt - t)/self.l_greedy.endt + self.l_greedy.end
        lr = np.max([lr, self.l_greedy.end])

        self.network.apply_update(lr, -self.wc)

    def perceive(self, reward, state, term, test=False):

        state = preproc(state, (84, 84))

        if self.max_reward:
            reward = np.min([reward, self.max_reward])

        if self.min_reward:
            reward = np.max([reward, self.min_reward])

        if self.rescale_r:
            self.r_max = np.max([self.r_max, reward])

        self.transitions.add_recent_state(state, term)

        # never called
        # curr_full_state = self.transitions.get_recent()

        if self.last_state is not None and not test:
            self.transitions.add(
                self.last_state, self.last_action, reward,
                self.last_term)

        # if self.num_steps == self.l_start + 1 and not test:
        #     self.sample_validation_data()

        curr_state = self.transitions.get_recent()
        curr_state = np.expand_dims(curr_state, axis=0)

        action_idx = 0
        if not term:
            action_idx = self.step(curr_state)

        self.transitions.add_recent_action(action_idx)

        if (self.num_steps > self.l_start and not test
                and self.num_steps % self.update_freq == 0):
            for i in range(self.n_replay):
                self.qLearnMinibatch()

        if not test:
            self.num_steps = self.num_steps + 1

        self.last_state = state
        self.last_action = action_idx
        self.last_term = term

        if self.target_q and self.num_steps % self.target_q == 1:
            logger.debug("Copy target_net weights")
            self.target_net.load_state_dict(self.network.state_dict())
            # self.target_net.set_weights(self.network.get_weights())

        if not term:
            return build_action_array(self.n_actions, action_idx)

        return [0] * self.n_actions
