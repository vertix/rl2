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

import pdb
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
        w.vision_range, w.remaining_action_cooldown_ticks]), dist


def EncodeMinion(m, me):
    dist = m.get_distance_to_unit(me)
    # TODO(handle neutral minions)
    m_type = 'm' if m.type == MinionType.ORC_WOODCUTTER else 'f'
    return np.array(EncodeType(m_type) + [
        m.life, m.max_life, 0., 0., m.speed_x, m.speed_y, m.angle,
        m.x - me.x, m.y - me.y, dist, 0.,
        m.vision_range, m.remaining_action_cooldown_ticks]), dist


def EncodeBuilding(b, me):
    dist = b.get_distance_to_unit(me)
    b_type = 't' if b.type == BuildingType.GUARDIAN_TOWER else 'b'
    return np.array(EncodeType(b_type) + [
        b.life, b.max_life, 0., 0., b.speed_x, b.speed_y, b.angle,
        b.x - me.x, b.y - me.y, dist, b.attack_range,
        b.vision_range, b.remaining_action_cooldown_ticks]), dist


DEFAULT_OTHER_STATE = np.array(EncodeType('') + [
    0., 0., # Life, max life
    0., 0., # Mana, max mana
    0., 0., 0., # Speed x, y, and angle
    0., 0., 0., # Delta x, Delta y and distance
    0., 0., 0., # Cast and vision ranges, remaining_cooldown
])


def NormalizeObjects(objs):
    objs = sorted(objs, key=lambda x: x[1])[:MAX_TARGETS_NUM]
    objs = [v for v, _ in objs]
    objs.extend([DEFAULT_OTHER_STATE] * (MAX_TARGETS_NUM - len(objs)))
    return np.hstack(objs)


MAX_TARGETS_NUM = 5

class State(object):
    def __init__(self, me, world, game):
        self.me = me
        self.world = world
        self.game = game
    
    def Get(self, fields):
        result = {}
        if fields is None or 'my_base_state' in fields:
            result['my_base_state'] = np.array(
            [self.me.life, self.me.max_life, self.me.mana,
             self.me.max_mana, self.me.angle, self.me.speed_x, self.me.speed_y,
             self.me.remaining_action_cooldown_ticks])
        if fields is None or 'other_base_state' in fields:
            MAX_RADIUS = 1200.
            frie_objs, host_obj = [], []
            for w in self.world.wizards:
                objs = frie_objs if w.faction == self.me.faction else host_obj
                objs.append(EncodeWizard(w, self.me))
            for m in self.world.minions:
                if m.faction in [Faction.NEUTRAL, Faction.OTHER]:
                    continue  # TODO(vertix): HANDLE NEUTRAL MINIONS

                objs = frie_objs if m.faction == self.me.faction else host_obj
                objs.append(EncodeMinion(m, self.me))

            for b in self.world.buildings:
                objs = frie_objs if b.faction == self.me.faction else host_obj
                objs.append(EncodeBuilding(b, self.me))
            
            result['other_base_state'] = np.hstack([NormalizeObjects(frie_objs), NormalizeObjects(host_obj)])

        return result


class Action(object):
    # NOOP, FLEE, PROCEED, ATTACK@X
    def __init__(self, move):
        self.move = move


class Policy(object):
    def Act(self, state):
        move = Move()
        
        move.speed = state.game.wizard_forward_speed
        move.strafe_speed = state.game.wizard_strafe_speed
        move.turn = state.game.wizard_max_turn_angle
        move.action = ActionType.MAGIC_MISSILE

        return Action(move)


class MyStrategy:
    def __init__(self):
        self.policy = Policy()
        self.last_score = 0.
        self.initialized = False
        self.last_state = {}
        self.last_action = -1
    
    def EncodeState(self, me, world, game):
        return State(me, world, game)

    def ImplementAction(self, action, move):
        for attr in ['strafe_speed'
                    ,'turn'
                    ,'action'
                    ,'cast_angle'
                    ,'min_cast_distance'
                    ,'max_cast_distance'
                    ,'status_target_id'
                    ,'skill_to_learn'
                    ,'messages']:
            setattr(move, attr, getattr(action.move, attr))

    
    def SaveExperience(self, s, a, r, s1):
        pass


    def move(self, me, world, game, move):
        """
        @type me: Wizard
        @type world: World
        @type game: Game
        @type move: Move
        """
        advance = Actions.AdvanceAction(game.map_size, LaneType.TOP)
        my_move = advance.Act(me, world, game)

        for attr in ['speed', 'strafe_speed', 'turn', 'action', 'cast_angle', 'min_cast_distance',
                     'max_cast_distance', 'status_target_id', 'skill_to_learn', 'messages']:
            setattr(move, attr, getattr(my_move, attr))

        print move.speed, move.strafe_speed, move.turn

        return

        reward = world.get_my_player().score - self.last_score
        
        state = self.EncodeState(me, world, game)
        cur_state_dict = state.Get(None)
        if self.initialized:
            self.SaveExperience(self.last_state, self.last_action, reward, cur_state_dict)

        # print {k: v.shape for k, v in cur_state_dict.iteritems()}
        # print list(cur_state_dict['other_base_state'])

        action = self.policy.Act(state)
        self.ImplementAction(action, move)

        self.last_score = world.get_my_player().score
        self.last_state = cur_state_dict
        self.last_action = action
        self.initialized = True
