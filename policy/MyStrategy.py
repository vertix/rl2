import cPickle
import os
import random
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
        state = BatchNorm(state, self.vars, 'model/hidden1/BatchNorm')
        state = ReLu(state)

        state = np.matmul(state, self.vars['model/hidden2/weights:0'])
        state = BatchNorm(state, self.vars, 'model/hidden2/BatchNorm')
        state = ReLu(state)

        value = np.matmul(state, self.vars['model/val_hid/weights:0'])
        value = BatchNorm(value, self.vars, 'model/val_hid/BatchNorm')
        value = ReLu(value)
        value = np.matmul(value, self.vars['model/value/weights:0'])
        value += self.vars['model/value/biases:0']

        adv = np.matmul(state, self.vars['model/adv_hid/weights:0'])
        adv = BatchNorm(adv, self.vars, 'model/adv_hid/BatchNorm')
        adv = ReLu(adv)
        adv = np.matmul(adv, self.vars['model/advantage/weights:0'])
        adv += self.vars['model/advantage/biases:0']

        return value + (adv - adv.mean())

    def Select(self, state):
        value = self.Q(state.to_numpy())
        res = np.argmax(value)
        return res, value[res]


class DefaultPolicy(object):
    def __init__(self, lane):
	self.lane = lane
	self.target = 0
	self.ticks_on_target = 0

    def Act(self, state):
	if state.my_state.hp < 50:
            res = 0 + self.lane  # FLEE
	    self.ticks_on_target = 100
        elif state.enemy_states:
	    if self.ticks_on_target > 50:
		self.ticks_on_target = 0
		self.target = random.choice(
		    range(min(MAX_TARGETS_NUM, len(state.enemy_states))))
            res = 6 + self.target   # RANGE ATTACK CLOSEST
	    self.ticks_on_target += 1
        else:
	    self.ticks_on_target = 100
            res = 3 + self.lane  # ADVANCE
	print res
	return res

    def Stop(self):
	pass


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

        if np.random.rand() < epsilon or self.q is None:
            return np.random.choice(range(self.max_actions))

        # if self.q == None: # or np.random.rand() < 0.5:
        #     if state.my_state.hp < 50:
        #         # print 'FLEE'
        #         return 0  # FLEE
        #     elif state.enemy_states:
        #         return 2  # RANGE ATTACK CLOSEST
        #     else:
        #         # print 'ADVANCE'
        #         return 1  # ADVANCE
        # else:
        res, val = self.q.Select(state)
        action = (['FLEE_%s' % ln for ln in ['TOP', 'MIDDLE', 'BOTTOM']] +
                  ['ADVANCE_%s' % ln for ln in ['TOP', 'MIDDLE', 'BOTTOM']] +
                  ['ATTACK_%d' %i for i in range(1, MAX_TARGETS_NUM + 1)])[res]
        if action != self.last_action:
            self.last_action = action
            print '%s: %.2f' % (action, val)
        return res


NUM_ACTIONS = 6 + MAX_TARGETS_NUM
LANES = [LaneType.TOP, LaneType.MIDDLE, LaneType.BOTTOM]

class MyStrategy:
    def __init__(self):
        if len(sys.argv) > 1 and zmq:
            self.sock = zmq.Context().socket(zmq.REQ)
            self.sock.connect(sys.argv[1])
            print 'Connected to %s' % sys.argv[1]
        else:
            self.sock = None

        if len(sys.argv) > 2 and sys.argv[2] and sys.argv[2] != '0' and zmq:
            self.policy = RemotePolicy(sys.argv[2], NUM_ACTIONS)
        else:
            self.policy = DefaultPolicy(random.choice(LANES))

        self.last_score = 0.
        self.initialized = False
        self.last_state = {}
        self.last_action = -1
        self.last_tick = None

        self.exp = {'s':[], 'a': [], 'r': [], 's1': []}
        self.next_file_index = 0
        self.flee_actions = []
        self.advance_actions = []

    def stop(self):
        self.SaveExperience(self.last_state, self.last_action, 0, None)
        self.policy.Stop()

    def SaveExperience(self, s, a, r, s1):
        if not self.sock:
            return

        data = {
            's': s.to_numpy(),
            'a': a,
            'r': r,
            's1': s1.to_numpy() if s1 else None
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
        if not self.flee_actions:
            for lane in LANES:
                self.flee_actions.append(Actions.FleeAction(game.map_size, lane))
                self.advance_actions.append(Actions.AdvanceAction(game.map_size, lane))

        state = State.WorldState(me, world, game)
        noop = Actions.NoOpAction()

        targets = [enemy.unit for enemy in state.enemy_states][:MAX_TARGETS_NUM]
        actions = (self.flee_actions + self.advance_actions +
                   [Actions.RangedAttack(
                       game.map_size,
                       random.choice([LaneType.TOP, LaneType.MIDDLE, LaneType.BOTTOM]), t)
                    for t in targets] +
                   [noop] * (MAX_TARGETS_NUM - len(targets)))

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
