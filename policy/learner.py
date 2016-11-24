import random

import numpy as np


class ExperienceBuffer(object):
    """Simple experience buffer"""
    def __init__(self, buffer_size=1 << 16, gamma=0.995):
        self.ss, self.aa, self.rr, self.ss1, self.gg = None, None, None, None, None
        self.buffer_size = buffer_size
        self.inserted = 0
        self.index = []
        self.gamma = gamma

    def add(self, s, a, r, s1):
        if self.ss is None:
            # Initialize
            state_size = len(s)
            self.ss = np.zeros((state_size, self.buffer_size))
            self.aa = np.zeros(self.buffer_size, dtype=np.int16)
            self.ss1 = np.zeros((state_size, self.buffer_size))
            self.rr = np.zeros(self.buffer_size)
            self.gg = np.zeros(self.buffer_size)

        cur_index = self.inserted % self.buffer_size
        self.ss[:, cur_index] = s
        self.aa[cur_index] = a
        self.rr[cur_index] = r
        if s1 is not None:
            self.ss1[:, cur_index] = s1
            self.gg[cur_index] = self.gamma
        else:
            self.ss1[:, cur_index] = s
            self.gg[cur_index] = 0.

        if len(self.index) < self.buffer_size:
            self.index.append(self.inserted)
        self.inserted += 1

    @property
    def state_size(self):
        return None if self.ss is None else self.ss.shape[0]

    def sample(self, size):
        if size > self.inserted:
            return None, None, None, None, None

        indexes = random.sample(self.index, size)

        return (np.transpose(self.ss[:,indexes]), self.aa[indexes], self.rr[indexes],
                np.transpose(self.ss1[:, indexes]), self.gg[indexes])


class WeightedExperienceBuffer(object):
    def __init__(self, buffer_size=1 << 16, gamma=0.995):
        self.ss, self.aa, self.rr, self.ss1, self.gg = None, None, None, None, None
        self.buffer_size = buffer_size
        self.inserted = 0
        self.tree_size = buffer_size << 1
        # root is 1
        self.weight_sums = [0.0] * self.tree_size
        self.gamma = gamma

    def update_up(self, index):
        self.weight_sums[index] = self.weight_sums[index << 1] + self.weight_sums[(index << 1) + 1]
        if index > 1:
            self.update_up(index >> 1)

    def index_in_tree(self, buffer_index):
        return buffer_index + self.buffer_size

    def index_in_buffer(self, tree_index):
        return tree_index - self.buffer_size

    def tree_update(self, buffer_index, new_weight):
        index = self.index_in_tree(buffer_index)
        self.weight_sums[index] = new_weight
        self.update_up(index >> 1)

    def add(self, s, a, r, s1, weight):
        if self.ss is None:
            # Initialize
            state_size = len(s)
            self.ss = np.zeros((state_size, self.buffer_size))
            self.aa = np.zeros(self.buffer_size, dtype=np.int16)
            self.ss1 = np.zeros((state_size, self.buffer_size))
            self.rr = np.zeros(self.buffer_size)
            self.gg = np.zeros(self.buffer_size)

        cur_index = self.inserted % self.buffer_size
        self.ss[:, cur_index] = s
        self.aa[cur_index] = a
        self.rr[cur_index] = r
        if s1 is not None:
            self.ss1[:, cur_index] = s1
            self.gg[cur_index] = self.gamma
        else:
            self.ss1[:, cur_index] = s
            self.gg[cur_index] = 0.

        self.inserted += 1

        self.tree_update(cur_index, weight)

    @property
    def state_size(self):
        return None if self.ss is None else self.ss.shape[0]
    
    def find_sum(self, node, sum):
        if node >= self.buffer_size:
            return self.index_in_buffer(node)
        left = node << 1
        left_sum = self.weight_sums[left]
        if sum < left_sum :
            return self.find_sum(left, sum)
        else:
            return self.find_sum(left + 1, sum - left_sum)
    
    def sample_indexes(self, size):
        total_weight = self.weight_sums[1]
        indexes = []
        for i in xrange(size):
            search = np.random.random() * total_weight
            indexes.append(self.find_sum(1, search))
        return indexes

    def sample(self, size):
        if size > self.inserted:
            return None, None, None, None, None, None

        indexes = self.sample_indexes(size)

        return (indexes, 
                np.transpose(self.ss[:,indexes]), self.aa[indexes], self.rr[indexes],
                np.transpose(self.ss1[:, indexes]), self.gg[indexes])