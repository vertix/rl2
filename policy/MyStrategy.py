import cPickle
import re
import sys
import os

from model.ActionType import ActionType
from model.BuildingType import BuildingType
from model.LaneType import LaneType
from model.Game import Game
from model.Faction import Faction
from model.MinionType import MinionType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World

import State

try:
    import zmq
except ImportError:
    print "ZMQ is not availabe"
    zmq = None

import Actions

import numpy as np

def EncodeFaction(their, mine):
    # [1, 0, 0] if its our unit
    # [0, 1, 0] if its enemy
    # [0, 0, 1] if its neutral
    if their == mine:
        return [1, 0, 0]
    elif their == Faction.NEUTRAL or their == Faction.OTHER:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def EncodeType(obj):
    result = [0] * 5
    if obj == 'w':
        result[0] = 1.
    elif obj == 'm':
        result[1] = 1.
    elif obj == 'f':
        result[2] = 1.
    elif obj == 't':
        result[3] = 1.
    elif obj == 'b':
        result[4] = 1.
    return result


def EncodeWizard(w, me):
    dist = w.get_distance_to_unit(me)
    return np.array(EncodeType('w') + [
        w.life, w.max_life, w.mana, w.max_mana, w.speed_x, w.speed_y, w.angle,
        w.x - me.x, w.y - me.y, dist, w.cast_range,
        w.vision_range, w.remaining_action_cooldown_ticks]), dist, w


def EncodeMinion(m, me):
    dist = m.get_distance_to_unit(me)
    # TODO(handle neutral minions)
    m_type = 'm' if m.type == MinionType.ORC_WOODCUTTER else 'f'
    return np.array(EncodeType(m_type) + [
        m.life, m.max_life, 0., 0., m.speed_x, m.speed_y, m.angle,
        m.x - me.x, m.y - me.y, dist, 0.,
        m.vision_range, m.remaining_action_cooldown_ticks]), dist, m


def EncodeBuilding(b, me):
    dist = b.get_distance_to_unit(me)
    b_type = 't' if b.type == BuildingType.GUARDIAN_TOWER else 'b'
    return np.array(EncodeType(b_type) + [
        b.life, b.max_life, 0., 0., b.speed_x, b.speed_y, b.angle,
        b.x - me.x, b.y - me.y, dist, b.attack_range,
        b.vision_range, b.remaining_action_cooldown_ticks]), dist, b


DEFAULT_OTHER_STATE = np.array([
    0., 0., # Life, max life
    0., 0., # Mana, max mana
    0., 0., 0., # Speed x, y, and angle
    0., 0., 0., # Delta x, Delta y and distance
    0., 0., 0., # Cast and vision ranges, remaining_cooldown
] + EncodeType(''))


def NormalizeObjects(objs):
    objs = sorted(objs, key=lambda x: x[1])[:MAX_TARGETS_NUM]
    objs = [v for v, _, _ in objs]
    objs.extend([DEFAULT_OTHER_STATE] * (MAX_TARGETS_NUM - len(objs)))
    return np.hstack(objs)


MAX_TARGETS_NUM = 5


def ReLu(x):
    return np.maximum(x, 0)


class QFunction(object):
    def __init__(self, network_vars):
        self.vars = network_vars

    def Q(self, state):
        state = np.matmul(state, self.vars['model/hidden1/weights:0'])
        state += self.vars['model/hidden1/biases:0']
        state = ReLu(state)

        state = np.matmul(state, self.vars['model/hidden2/weights:0'])
        state += self.vars['model/hidden2/biases:0']
        state = ReLu(state)

        state = np.matmul(state, self.vars['model/output/weights:0'])
        state += self.vars['model/output/biases:0']
        return state

    def Select(self, state):
        return np.argmax(self.Q(state.to_numpy()))


class RemotePolicy(object):
    def __init__(self, address, max_actions):
        self.active = True
        if address and zmq:
            self.sock = zmq.Context().socket(zmq.SUB)
            self.sock.setsockopt(zmq.SUBSCRIBE, "")
            self.sock.connect(address)
            import threading
            # TODO(vertix): COLLECT THE THREAD!!!
            self.thread = threading.Thread(target=self.Listen)
            self.thread.start()
        else:
            self.sock = None

        self.q = None
        self.steps = 0
        self.max_actions = max_actions

    def Listen(self):
        while self.active:
            self.q = QFunction(self.sock.recv_pyobj())
            print 'Recieved coeff'

    def Act(self, state):
        epsilon = 0.5 / (1 + self.steps / 1000.)
        self.steps += 1

        if np.random.rand() < epsilon:
            return np.random.choice(range(self.max_actions))

        if self.q == None or np.random.rand() < 0.5:
            if state.my_state.hp < 50:
                # print 'FLEE'
                return 0  # FLEE
            elif state.enemy_states:
                return 2  # RANGE ATTACK CLOSEST
            else:
                # print 'ADVANCE'
                return 1  # ADVANCE
        else:
            return self.q.Select(state)


NUM_ACTIONS = 2 + MAX_TARGETS_NUM

class MyStrategy:
    def __init__(self):
        if len(sys.argv) > 1 and zmq:
            self.sock = zmq.Context().socket(zmq.REQ)
            self.sock.connect(sys.argv[1])
            print 'Connected to %s' % sys.argv[1]
        else:
            self.sock = None

        if len(sys.argv) > 2 and zmq:
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

    def __del__(self):
        self.policy.thread.join(0.1)

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
