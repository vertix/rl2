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

from Analysis import PickTarget
from Analysis import HistoricStateTracker

from Geometry import GetLanes

import Actions
import State


try:
    import zmq
except ImportError:
    print "ZMQ is not availabe"
    zmq = None

debug = None
try:
    from debug_client import DebugClient
except:
    debug = None
else:
    pass
    # debug = DebugClient()


MAX_TARGETS_NUM = 5


def ReLu(x):
    return np.maximum(x, 0)


def BatchNorm(state, network_vars, key):
    eps = 0.001
    inv = 1.0 / np.sqrt(network_vars[key + '/moving_variance:0'] + eps)

    return state * inv + (network_vars[key + '/beta:0'] - network_vars[key + '/moving_mean:0'] * inv)


def Softmax(state):
    state -= np.max(state)
    e = np.exp(state)
    return e / np.sum(e)


class QFunction(object):
    def __init__(self, network_vars):
        self.vars = network_vars

    def Q(self, state):
        state = np.matmul(state, self.vars['model/hidden1/weights:0'])
        state += self.vars['model/hidden1/biases:0']
        # state = BatchNorm(state, self.vars, 'model/hidden1/BatchNorm')
        state = ReLu(state)

        state = np.matmul(state, self.vars['model/hidden2/weights:0'])
        state += self.vars['model/hidden2/biases:0']
        # state = BatchNorm(state, self.vars, 'model/hidden2/BatchNorm')
        state = ReLu(state)

        value = np.matmul(state, self.vars['model/val_hid/weights:0'])
        value += self.vars['model/val_hid/biases:0']
        # value = BatchNorm(value, self.vars, 'model/val_hid/BatchNorm')
        value = ReLu(value)
        value = np.matmul(value, self.vars['model/value/weights:0'])
        value += self.vars['model/value/biases:0']

        adv = np.matmul(state, self.vars['model/adv_hid/weights:0'])
        adv += self.vars['model/adv_hid/biases:0']
        # adv = BatchNorm(adv, self.vars, 'model/adv_hid/BatchNorm')
        adv = ReLu(adv)
        adv = np.matmul(adv, self.vars['model/advantage/weights:0'])
        adv += self.vars['model/advantage/biases:0']

        return value + (adv - adv.mean())

    def Select(self, state):
        value = self.Q(state.to_numpy())
        res = np.argmax(value)
        return res, value[res]

MAX_ATTACK_DISTANCE = 1000


class VarsListener(object):
    def __init__(self, address, setter):
        self.setter = setter

        self.sock = zmq.Context().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, "")
        self.sock.connect(address)

        import threading
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.Listen)
        self.thread.start()

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
                self.setter(evts[0][0].recv_pyobj())
        print 'Exitting...'


class NNPolicy(object):
    def __init__(self, network_vars, max_actions):
        self.vars = network_vars
        self.actions = range(max_actions)

    def UpdateVars(self, new_vars):
        self.vars = new_vars

    def Logits(self, state):
        state = np.matmul(state, self.vars['common/hidden1/weights:0'])
        state += self.vars['common/hidden1/biases:0']
        state = ReLu(state)

        state = np.matmul(state, self.vars['common/hidden2/weights:0'])
        state += self.vars['common/hidden2/biases:0']
        state = ReLu(state)

        logits = np.matmul(state, self.vars['policy/policy/weights:0'])
        logits += self.vars['policy/policy/biases:0']
        return logits

    def Softmax(self, state):
        logits = self.Logits(state)
        return Softmax(logits)

    def Sample(self, state):
        if self.vars is None:
            return random.choice(self.actions)

        sm = self.Softmax(state)
        if self.actions is None:
            self.actions = range(len(sm))
        return np.random.choice(self.actions, p=sm)

    def Act(self, state):
        return self.Sample(state.to_numpy())


class QPolicy(object):
    def __init__(self, max_actions):
        self.q = None
        self.steps = 0
        self.max_actions = max_actions

    def UpdateVars(self, new_vars):
        self.q = QFunction(new_vars)

    def Act(self, state):
        epsilon = 0.5 / (1 + self.steps / 1000.)
        self.steps += 1

        if np.random.rand() < epsilon or self.q is None:
            return np.random.randint(0, self.max_actions)

        res, _ = self.q.Select(state)
        return res


class DefaultPolicy(object):
    def Act(self, state):
        if state.my_state.hp - state.my_state.unit.max_life * 0.63 < state.my_state.aggro:
            res = 0 # FLEE
        else:
            u = PickTarget(state.my_state.me, ActionType.MAGIC_MISSILE, state,
                           radius=MAX_ATTACK_DISTANCE, lane=state.my_state.lane)
            if u:
                e_ids = [e.unit.id for e in state.enemy_states[:MAX_TARGETS_NUM]]
                idx = e_ids.index(u.id) if u.id in e_ids else 0
                res = 2 + idx   # ATTACK
            else:
                res = 1 # ADVANCE
        return res

    def UpdateVars(self, new_vars):
        pass


NUM_ACTIONS = 2 + MAX_TARGETS_NUM
LANES = [LaneType.TOP, LaneType.MIDDLE, LaneType.BOTTOM]
GAMMA = 0.995
Q_N_STEPS = 20


class MyStrategy:
    def __init__(self, args=None):
        self.args = args
        if zmq and args and args.exp_socket:
            self.sock = zmq.Context().socket(zmq.REQ)
            addr = 'tcp://127.0.0.1:%d' % args.exp_socket
            self.sock.connect(addr)
            print 'Connected to %s' % addr
        else:
            self.sock = None

        self.listener = None
        if args and args.policy == 'q' and args.vars_socket:
            self.policy = QPolicy(NUM_ACTIONS)
        elif args and args.policy == 'nn':
            # with open('network') as f:
                # cPickle.load(f)
            self.policy = NNPolicy(None, NUM_ACTIONS)
        else:
            self.policy = DefaultPolicy()

        if self.args and self.args.vars_socket and zmq:
            self.listener = VarsListener('tcp://127.0.0.1:%d' % args.vars_socket,
                                         self.policy.UpdateVars)

        self.last_score = 0.
        self.initialized = False
        self.last_state = None
        self.last_action = -1
        self.last_tick = None
        self.num_deaths = 0

        self.exps = []
        self.next_file_index = 0
        self.flee_action = None
        self.advance_action = None

    def GetLane(self, me):
        if self.args and self.args.random_lane:
            return random.choice(LANES)

        if me.master:
            return LaneType.MIDDLE
        messages = me.messages
        for m in reversed(messages):
            if m.lane is not None:
                return m.lane
        return None

    def stop(self, final_score=None):
        r = 0.
        if final_score is not None:
            r = final_score - self.last_score
            print 'Final score is %d' % final_score
            print 'Final reward is %d' % r

        self.SaveExperience(self.last_state, self.last_action, r, None, 0.)
        if self.sock and not isinstance(self.policy, DefaultPolicy):
            self.sock.send_pyobj({
                'type':'stat',
                'data': {
                    'Stats/Score': self.last_score + r,
                    'Stats/Length': self.last_tick,
                    'Stats/Num Deaths': self.num_deaths
                }})
            print 'Saving stats'
        if self.listener:
            self.listener.Stop()

    def SaveExperience(self, s, a, r, s1, gamma):
        if not self.sock or (s is None):
            return

        s1 = s1.to_numpy() if s1 else s.to_numpy()
        self.exps.append({
            's': s.to_numpy(),
            'a': a,
            'r': r,
            's1': s1,
            'g': gamma
        })

        if s1 is None or len(self.exps) >= Q_N_STEPS:
            if self.args and self.args.n_step:
                rew = 0.
                g = 1.
                for exp in reversed(self.exps):
                    rew = exp['r'] + exp['g'] * rew
                    g *= exp['g']
                    exp['s1'] = s1
                    exp['r'] = rew
                    exp['g'] = g

            s = np.array([e['s'] for e in self.exps])
            a = np.array([e['a'] for e in self.exps], dtype=np.int32)
            r = np.array([e['r'] for e in self.exps])
            s1 = np.array([e['s1'] for e in self.exps])
            g = np.array([e['g'] for e in self.exps])

            self.sock.send_pyobj({'type': 'exp', 'data': {
                's': s, 'a': a, 'r': r, 's1': s1, 'g': g
            }})
            if self.sock.recv() != "Ok":
                print "Error when sending experience"
            self.exps = []

    def move(self, me, world, game, move):
        """
        @type me: Wizard
        @type world: World
        @type game: Game
        @type move: Move
        """
        if debug:
            debug.post()
            debug.start()
        HistoricStateTracker.GetInstance(me, world).AddInvisibleBuildings(me, world, game)
        if world.tick_index < 10:
            l = self.GetLane(me)
            if l is not None:
                self.lane = l
                self.flee_action = Actions.FleeAction(game.map_size, self.lane)
                self.advance_action = Actions.AdvanceAction(game.map_size, self.lane)

        if self.flee_action is None:
            self.lane = np.random.choice(LANES)
            self.flee_action = Actions.FleeAction(game.map_size, self.lane)
            self.advance_action = Actions.AdvanceAction(game.map_size, self.lane)

        state = None
        state = State.WorldState(me, world, game, self.lane, self.last_state, debug)
        noop = Actions.NoOpAction()

        targets = [enemy.unit for enemy in state.enemy_states
                   if enemy.dist < 1000][:MAX_TARGETS_NUM]
        actions = ([self.flee_action, self.advance_action] +
                   [Actions.MeleeAttack(game.map_size, self.lane, t) for t in targets] +
                   [noop] * (MAX_TARGETS_NUM - len(targets)))

        a = self.policy.Act(state)
        if self.args and self.args.verbose and a != self.last_action:
            print actions[a].name

        reward = world.get_my_player().score - self.last_score
        gamma = GAMMA
        if self.last_tick and world.tick_index - self.last_tick > 1:
            gamma = GAMMA ** (world.tick_index - self.last_tick)
            self.num_deaths += 1

        if self.args and self.args.verbose and reward != 0:
            print 'REWARD: %.0f' % reward

        if self.initialized:
            self.SaveExperience(self.last_state, self.last_action, reward, state, gamma)

        my_move = actions[a].Act(me, state)
        for attr in ['speed', 'strafe_speed', 'turn', 'action', 'cast_angle', 'min_cast_distance',
                     'max_cast_distance', 'status_target_id', 'skill_to_learn', 'messages']:
            setattr(move, attr, getattr(my_move, attr))

        self.last_score = world.get_my_player().score
        self.last_state = state
        self.last_action = a
        self.initialized = True
        self.last_tick = world.tick_index
        if debug:
            debug.stop()
        # print world.tick_index
