import random
import time

import numpy as np
import tensorflow as tf


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
    def __init__(self, alpha, beta, max_weight, buffer_size=1<<16):
        self.ss, self.aa, self.rr, self.ss1, self.gg = None, None, None, None, None
        self.buffer_size = buffer_size
        self.inserted = 0
        self.tree_size = buffer_size << 1
        # root is 1
        self.weight_sums = np.zeros(self.tree_size)
        self.weight_min = np.ones(self.tree_size) * (max_weight ** alpha)
        self.max_weight = max_weight
        self.alpha = alpha
        self.beta = beta

    def update_up(self, index):
        self.weight_sums[index] = self.weight_sums[index << 1] + self.weight_sums[(index << 1) + 1]
        self.weight_min[index] = min(self.weight_min[index << 1], self.weight_min[(index << 1) + 1])
        if index > 1:
            self.update_up(index >> 1)

    def index_in_tree(self, buffer_index):
        return buffer_index + self.buffer_size

    def index_in_buffer(self, tree_index):
        return tree_index - self.buffer_size

    def tree_update(self, buffer_index, new_weight):
        index = self.index_in_tree(buffer_index)
        new_weight = min(new_weight + 0.01, self.max_weight) ** self.alpha

        self.weight_sums[index] = new_weight
        self.weight_min[index] = new_weight
        self.update_up(index >> 1)

    def add(self, s, a, r, s1, gamma, weight):
        if self.ss is None:
            # Initialize
            state_size = s.shape[1]
            self.ss = np.zeros((state_size, self.buffer_size))
            self.aa = np.zeros(self.buffer_size, dtype=np.int16)
            self.ss1 = np.zeros((state_size, self.buffer_size))
            self.rr = np.zeros(self.buffer_size)
            self.gg = np.zeros(self.buffer_size)

        indexes = []
        for _ in a:
            cur_index = self.inserted % self.buffer_size
            self.inserted += 1
            indexes.append(cur_index)

        self.ss[:, indexes] = s.transpose()
        self.aa[indexes] = a
        self.rr[indexes] = r
        self.ss1[:, indexes] = s1.transpose()
        self.gg[indexes] = gamma

        for idx in indexes:
            self.tree_update(idx, weight)

    @property
    def state_size(self):
        return None if self.ss is None else self.ss.shape[0]

    def find_sum(self, node, sum):
        if node >= self.buffer_size:
            return self.index_in_buffer(node)
        left = node << 1
        left_sum = self.weight_sums[left]
        if sum < left_sum:
            return self.find_sum(left, sum)
        else:
            return self.find_sum(left + 1, sum - left_sum)

    def sample_indexes(self, size):
        total_weight = self.weight_sums[1]
        indexes = np.zeros(size, dtype=np.int32)
        for i in xrange(size):
            search = np.random.random() * total_weight
            indexes[i] = self.find_sum(1, search)
        return indexes

    def sample(self, size):
        if size > self.inserted:
            return None, None, None, None, None, None, None

        indexes = self.sample_indexes(size)
        max_w = (self.weight_min[1] / self.weight_sums[1]) ** -self.beta
        w = (self.weight_sums[self.index_in_tree(indexes)] / self.weight_sums[1]) ** -self.beta

        return (indexes,
                np.transpose(self.ss[:, indexes]), self.aa[indexes], self.rr[indexes],
                np.transpose(self.ss1[:, indexes]), self.gg[indexes],
                w / max_w)


def HuberLoss(tensor, boundary):
    abs_x = tf.abs(tensor)
    delta = boundary
    quad = tf.minimum(abs_x, delta)
    lin = (abs_x - quad)
    return 0.5 * quad**2 + delta * lin

DEFAULT_OPTIONS = {
    'clip_grad': 3.,
    'learning_rate': 0.0001,
}

class SupervisedPolicyValue(object):
    """Class to learn policy and value function on EXISTING policy"""
    def __init__(self, build_networks, buf, options=DEFAULT_OPTIONS):
        self._options = options
        self.exp_buffer = buf
        with tf.device('/cpu:0'):
            self.state = tf.placeholder(tf.float32, shape=[None, self.exp_buffer.state_size],
                                        name='state')
            self.action = tf.placeholder(tf.int32, shape=[None], name='action')
            self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
            self.state1 = tf.placeholder(tf.float32, shape=[None, self.exp_buffer.state_size],
                                         name='state1')
            self.gamma = tf.placeholder(tf.float32, shape=[None], name='gamma')
            self.is_weights = tf.placeholder(tf.float32, shape=[None], name='is_weights')
            self.is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

            self.logits, self.baseline = build_networks(self.state,
                                                        is_training=self.is_training, reuse=False)
            _, self.baseline1 = build_networks(self.state1, is_training=False, reuse=True)
            self.tf_policy = tf.reshape(tf.multinomial(self.logits, 1), [])

            # Experimental
            self.rolled_baseline = tf.stop_gradient(self.reward + self.gamma * self.baseline1)
            self.advantage = self.rolled_baseline - self.baseline

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.action)
            self.policy_loss = tf.reduce_mean(self.cross_entropy)
            # For actor-critic this should look like:
            # self.policy_loss = tf.reduce_mean(
            #     tf.mul(self.cross_entropy, tf.stop_gradient(self.advantage)))

            self.value_loss = 0.5 * tf.reduce_mean(HuberLoss(self.advantage, 5))

            self.policy_entropy = tf.reduce_mean(-tf.nn.softmax(self.logits) * 
                                                  tf.nn.log_softmax(self.logits))

#             loss = self.value_loss
            loss = self.policy_loss + 0.25 * self.value_loss - 0.01 * self.policy_entropy

            self.optimizer = tf.train.AdamOptimizer(options['learning_rate'])
            grads = self.optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.VARIABLES))
            if 'clip_grad' in options:
                grads = [(tf.clip_by_norm(g, options['clip_grad']) if g is not None else None, v)
                         for g, v in grads]

            for grad, var in grads:
                tf.histogram_summary(var.name, var)
                if grad is not None:
                    tf.histogram_summary('{}/grad'.format(var.name), grad)            

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.apply_gradients(grads, self.global_step)
            
            tf.histogram_summary("Predicted baseline", self.baseline)
            tf.histogram_summary("TD error", self.advantage)
            tf.scalar_summary("Loss/Actor", self.policy_loss)
            tf.scalar_summary("Loss/Critic", self.value_loss)
            tf.scalar_summary("Loss/Entropy", self.policy_entropy)
            tf.scalar_summary("Loss/Total", loss)

            self.summary_op = tf.merge_all_summaries()

    def Init(self, sess, run_id):
        sess.run(tf.initialize_all_variables())
        self.writer = tf.train.SummaryWriter(
            '/Users/vertix/tf/tensorflow_logs/aicup/%s'  % run_id
#             '/media/vertix/UHDD/tmp/tensorflow_logs/aicup/%s' % run_id
        )
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES))
        self.last_start = time.time()
        self.cur_step = 0
        self.writer.add_graph(tf.get_default_graph())

    def Step(self, sess, batch_size=32):
        idx, ss, aa, rr, ss1, gg, ww = self.exp_buffer.sample(batch_size)
        if ss is None:
            return
        
        feed_dict = {self.state: ss, self.action: aa, self.reward: rr, self.state1:ss1,
                     self.gamma: gg, self.is_weights: ww,
                     self.is_training: True}

        if self.cur_step and self.cur_step % 100 != 0:
            self.cur_step, _ = sess.run(
                [self.global_step, self.train_op], feed_dict)
        else:
            self.cur_step, _, smr = sess.run(
                [self.global_step, self.train_op, self.summary_op], feed_dict)
            self.writer.add_summary(smr, self.cur_step)

        if self.cur_step % 20000 == 0:
            self.saver.save(sess, 'ac', global_step=self.global_step)
            if self.last_start is not None:
                self.writer.add_summary(
                    tf.Summary(
                        value=[tf.Summary.Value(
                            tag='Steps per sec',
                            simple_value=20000 / (time.time() - self.last_start))]),
                    self.cur_step)
            self.last_start = time.time()