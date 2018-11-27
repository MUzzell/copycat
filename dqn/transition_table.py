import numpy as np

import pdb

import logging
logger = logging.getLogger(__name__)


def reverse_pop(a):
    a.reverse()
    a.pop()
    a.reverse()
    return a


class TransitionTable(object):

    def __init__(self, *args, **kwargs):

        self.state_dim = kwargs.get('state_dim')
        self.act_dim = kwargs.get('act_dim')
        self.n_actions = kwargs.get('n_actions')
        self.hist_len = kwargs.get('hist_len')

        assert self.state_dim and self.n_actions and self.hist_len

        self.max_size = kwargs.get('max_size', 1024^2)
        self.buffer_size = kwargs.get('buffer_size', 1024)

        self.hist_type = kwargs.get('hist_type', 'linear')

        if self.hist_type != 'linear':
            raise NotImplemented("hist_type != linear")

        self.hist_spacing = kwargs.get('hist_spacing', 1)

        self.zero_frames = kwargs.get('zero_frames', 1)
        self.non_term_prob = kwargs.get('non_term_prob', 1)
        self.non_event_prob = kwargs.get('non_event_prob', 1)

        self.num_entries = 0
        self.insert_index = 0

        self.recent_mem_size = self.hist_spacing * self.hist_len

        self.hist_ind = np.zeros(self.hist_len, dtype=np.uint32)

        for i in range(self.hist_len):
            self.hist_ind[i] = (i) * self.hist_spacing

        self.s = np.zeros((self.max_size, self.state_dim), dtype=np.uint8)
        self.a = np.zeros(self.max_size, dtype=np.uint32)
        self.r = np.zeros(self.max_size, dtype=np.uint32)
        self.t = np.zeros(self.max_size, dtype=np.uint8)

        self.action_encodings = np.identity(self.n_actions)

        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        #self.recent_s = np.zeros((self.hist_len, ) + self.state_dim, dtype=np.uint8)
        #self.recent_a = np.zeros(self.hist_len, dtype=np.uint32)
        #self.recent_t = np.zeros(self.hist_len, dtype=np.uint8)

        s_size = self.state_dim * self.hist_len
        self.buf_a = np.zeros(self.buffer_size)
        self.buf_r = np.zeros(self.buffer_size)
        self.buf_term = np.zeros(self.buffer_size)

        self.buf_s = np.zeros((self.buffer_size, s_size), dtype=np.uint8)
        self.buf_s2 = np.zeros((self.buffer_size, s_size), dtype=np.uint8)

        self.buf_ind = None

    def reset():
        self.num_entries = 0
        self.insert_index = 0

    @property
    def size(self):
        return self.num_entries

    @property
    def empty(self):
        return self.num_entries == 0

    def fill_bufer(self):
        assert self.num_entries >= self.buffer_size

        self.buf_ind = 0
        for buf_ind in range(self.buffer_size):
            s, a, r, s2, term = self.sample_one(1)
            self.buf_s[buf_ind] = np.copy(
                s.reshape(self.buf_s.shape[1])
            )
            self.buf_a[buf_ind] = np.copy(a)
            self.buf_r[buf_ind] = r
            self.buf_s2[buf_ind] = np.copy(
                s2.reshape(self.buf_s2.shape[1])
            )
            self.buf_term[buf_ind] = term

        self.buf_s = self.buf_s / 255
        self.buf_s2 = self.buf_s / 255

    def sample_one(self, i):
        assert self.num_entries > 1

        index = -1
        valid = False

        while not valid:
            index = np.random.randint(1, self.num_entries-self.recent_mem_size)

            if self.t[index+self.recent_mem_size-1] == 0:
                valid = True

            if (
              self.non_term_prob < 1 and
              self.t[index+self.recent_mem_size] == 0 and
              np.random.uniform() > self.non_term_prob):
                valid = False

            if (
              self.non_event_prob < 1 and
              self.t[index+self.recent_mem_size] == 1 and
              self.r[index+self.recent_mem_size-1] == 0 and
              np.random.uniform() > self.non_term_prob):
                valid = False

        return self[i]

    def sample(self, batch_size=1):
        assert batch_size < self.buffer_size

        if (
          not self.buf_ind or
          self.buf_ind + batch_size-1 > self.buffer_size):
            logger.debug("Filling buffer")
            self.fill_bufer()

        idx = self.buf_ind

        logger.debug("Sampling from buffer %d -> %d",
                     idx, idx + batch_size)

        self.buf_ind = self.buf_ind + batch_size

        return (self.buf_s[idx:idx+batch_size],
                self.buf_a[idx:idx+batch_size],
                self.buf_r[idx:idx+batch_size],
                self.buf_s2[idx:idx+batch_size],
                self.buf_term[idx:idx+batch_size])

    def concat_frames(self, idx, use_recent=False):
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        full_state = np.resize(s[0], (self.hist_len,)+s[0].shape)

        zero_out = False
        ep_start = self.hist_len

        for i in range(self.hist_len-2, -1, -1):
            if not zero_out:
                for j in range(idx+self.hist_ind[i]-1, idx+self.hist_ind[i+1]-2):
                    if t[j] == 1:
                        zero_out = True

            if zero_out:
                full_state[i] = 0
            else:
                ep_start = i

        if self.zero_frames == 0:
            ep_start = 0

        for i in range(ep_start, self.hist_len):
            full_state[i] = np.copy(s[idx + self.hist_ind[i]-1])

        return full_state

    def concat_actions(self, idx, use_recent=False):

        act_hist = np.array((self.hist_len, self.n_actions), dtype=np.float)

        if use_recent:
            a, t = self.recent_a, self.recent_t
        else:
            a, t = self.a, self.t

        zero_out = False
        ep_start = self.hist_len

        for i in range(self.hist_len-1, -1, -1):
            if not zero_out:
                for j in range(idx+self.hist_ind[i]-1,idx+self.hist_ind[i+1]-2):
                    if t[j] == 1:
                        zero_out = True

            if zero_out:
                act_hist[i] = 0
            else:
                ep_start = i

        if self.zero_frames == 0:
            ep_start = 1

        for i in range(ep_start, self.hist_len):
            act_hist[i] = np.copy(self.action_encodings[a[idx + self.hist_ind[i]-1]])

        return act_hist

    def get_recent(self):
        return self.concat_frames(1, True) / 255

    def __getitem__(self, idx):
        s = self.concat_frames(idx)
        s2 = self.concat_frames(idx+1)
        ar_idx = idx + self.recent_mem_size-1

        return s, self.a[ar_idx], self.r[ar_idx], s2, self.t[ar_idx+1]

    def add(self, s, a, r, term):
        assert s is not None, "State cannot be null"
        assert a is not None, "Action cannot be null"
        assert r is not None, "Reward cannot be null"

        if self.num_entries < self.max_size:
            self.num_entries += 1

        self.insert_index += 1

        if self.insert_index >= self.max_size:
            logger.debug("Resetting insert_index")
            self.insert_index = 0

        self.s[self.insert_index, :] = np.copy(s.reshape(self.state_dim)) # * 255
        self.a[self.insert_index] = a
        self.r[self.insert_index] = r

        if term:
            self.t[self.insert_index] = 1
        else:
            self.t[self.insert_index] = 0

    def add_recent_state(self, s, term):
        s = np.copy(s) # * 255

        if len(self.recent_s) == 0:
            for i in range(self.recent_mem_size):
                s_copy = np.copy(s)
                s_copy.fill(0)
                self.recent_s.append(s_copy)
                self.recent_t.append(1)

        self.recent_s.append(s)
        self.recent_t.append(1 if term else 0)

        if len(self.recent_s) > self.recent_mem_size:
            reverse_pop(self.recent_s)
            reverse_pop(self.recent_t)

    def add_recent_action(self, a):
        if len(self.recent_a) == 0:
            for i in range(self.recent_mem_size):
                self.recent_a.append(1)

        self.recent_a.append(a)

        if len(self.recent_a) > self.recent_mem_size:
            reverse_pop(self.recent_a)
