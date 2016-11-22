import cPickle
import os
import re
import sys
import time

import numpy as np

from model.ActionType import ActionType
from model.BuildingType import BuildingType
from model.LaneType import LaneType
from model.Game import Game
from model.Faction import Faction
from model.MinionType import MinionType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World

import Actions
import State

try:
    import zmq
except ImportError:
    print "ZMQ is not availabe"
    zmq = None


MAX_TARGETS_NUM = 5


def ReLu(x):
    return np.maximum(x, 0)


def BatchNorm(state, network_vars, key):
    eps = 0.001
    inv = 1.0 / np.sqrt(network_vars[key + '/moving_variance:0'] + eps)

    return state * inv + (network_vars[key + '/beta:0'] - network_vars[key + '/moving_mean:0'] * inv)


class QFunction(object):
    def __init__(self, network_vars):
        self.vars = network_vars

    def Q(self, state):
        state = np.matmul(state, self.vars['model/hidden1/weights:0'])
        # state += self.vars['model/hidden1/biases:0']
        state = BatchNorm(state, self.vars, 'model/hidden1/BatchNorm')
        state = ReLu(state)

        state = np.matmul(state, self.vars['model/hidden2/weights:0'])
        # state += self.vars['model/hidden2/biases:0']
        state = BatchNorm(state, self.vars, 'model/hidden2/BatchNorm')
        state = ReLu(state)

        value = np.matmul(state, self.vars['model/val_hid/weights:0'])
        value = BatchNorm(value, self.vars, 'model/val_hid/BatchNorm')
        # value += self.vars['model/val_hid/biases:0']
        value = ReLu(value)
        value = np.matmul(value, self.vars['model/value/weights:0'])
        value += self.vars['model/value/biases:0']

        adv = np.matmul(state, self.vars['model/adv_hid/weights:0'])
        adv = BatchNorm(adv, self.vars, 'model/adv_hid/BatchNorm')
        # adv += self.vars['model/adv_hid/biases:0']
        adv = ReLu(adv)
        adv = np.matmul(adv, self.vars['model/advantage/weights:0'])
        adv += self.vars['model/advantage/biases:0']

        return value + (adv - adv.mean())

    def Select(self, state):
        value = self.Q(state.to_numpy())
        res = np.argmax(value)
        return res, value[res]


class RemotePolicy(object):
    def __init__(self, address, max_actions):
        if address and zmq:
            self.sock = zmq.Context().socket(zmq.SUB)
            self.sock.setsockopt(zmq.SUBSCRIBE, "")
            self.sock.connect(address)
            import threading
            self._stop = threading.Event()
            self.thread = threading.Thread(target=self.Listen)
            self.thread.start()
        else:
            self.sock = None
            self._stop = None

        self.q = None
        self.steps = 0
        self.max_actions = max_actions
        self.last_action = None

    def Stop(self):
        if self._stop:
            self._stop.set()
            self.thread.join()

    def Listen(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)

        while not self._stop.isSet():
            evts = poller.poll(1000)
            if evts:
                self.q = QFunction(evts[0][0].recv_pyobj())
                print 'Recieved coeff'
        print 'Exitting...'

    def Act(self, state):
        epsilon = 0.5 / (1 + self.steps / 1000.)
        self.steps += 1

        if zmq and (np.random.rand() < epsilon):
            return np.random.choice(range(self.max_actions))

        if self.q == None: # or np.random.rand() < 0.5:
            if state.my_state.hp < 50:
                # print 'FLEE'
                return 0  # FLEE
            elif state.enemy_states:
                return 2  # RANGE ATTACK CLOSEST
            else:
                # print 'ADVANCE'
                return 1  # ADVANCE
        else:
            res, val = self.q.Select(state)
            action = (['FLEE', 'ADVANCE'] +
                      ['ATTACK_%d' %i for i in range(1, MAX_TARGETS_NUM + 1)])[res]
            if action != self.last_action:
                self.last_action = action
                print '%s: %.2f' % (action, val)
            return res


NUM_ACTIONS = 2 + MAX_TARGETS_NUM

class MyStrategy:
    def __init__(self):
        if zmq and len(sys.argv) > 1:
            self.sock = zmq.Context().socket(zmq.REQ)
            self.sock.connect(sys.argv[1])
            print 'Connected to %s' % sys.argv[1]
        else:
            self.sock = None

        if zmq and len(sys.argv) > 2:
            self.policy = RemotePolicy(sys.argv[2], NUM_ACTIONS)
        else:
            self.policy = RemotePolicy(None, NUM_ACTIONS)

        self.last_score = 0.
        self.initialized = False
        self.last_state = {}
        self.last_action = -1
        self.last_tick = None

        self.exp = {'s':[], 'a': [], 'r': [], 's1': []}
        self.next_file_index = 0
        self.flee_action = None
        self.advance_action = None
        self.lane = None

    def stop(self):
        self.policy.Stop()

    def SaveExperience(self, s, a, r, s1):
        if not self.sock:
            return

        data = {
            's': s.to_numpy(),
            'a': a,
            'r': r,
            's1': s1.to_numpy()
        }
        self.sock.send_pyobj(data)
        if self.sock.recv() != "Ok":
            print "Error when sending experience"

    def move(self, me, world, game, move):
        """
        @type me: Wizard
        @type world: World
        @type game: Game
        @type move: Move
        """
        if self.flee_action is None:
            if len(sys.argv) > 3:
                self.lane = int(sys.argv[3])
            else:
                self.lane = LaneType.TOP
            self.flee_action = Actions.FleeAction(game.map_size, self.lane)
            self.advance_action = Actions.AdvanceAction(game.map_size, self.lane)

        lane = self.lane

        state = State.WorldState(me, world, game)

        targets = [enemy.unit for enemy in state.enemy_states][:MAX_TARGETS_NUM]
        actions = ([self.flee_action, self.advance_action] +
                   [Actions.RangedAttack(game.map_size, lane, t) for t in targets] +
                   [self.advance_action] * (MAX_TARGETS_NUM - len(targets)))

        a = self.policy.Act(state)
        reward = world.get_my_player().score - self.last_score
        if self.last_tick and world.tick_index - self.last_tick > 1:
            reward = -500

        if reward != 0:
            print 'REWARD: %.1f' % reward

        if self.initialized:
            self.SaveExperience(self.last_state, self.last_action, reward, state)

        my_move = actions[a].Act(me, world, game)
        for attr in ['speed', 'strafe_speed', 'turn', 'action', 'cast_angle', 'min_cast_distance',
                     'max_cast_distance', 'status_target_id', 'skill_to_learn', 'messages']:
            setattr(move, attr, getattr(my_move, attr))

        self.last_score = world.get_my_player().score
        self.last_state = state
        self.last_action = a
        self.initialized = True
        self.last_tick = world.tick_index
