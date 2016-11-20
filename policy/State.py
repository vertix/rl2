import math

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


class State(object):
    """Base class for state"""
    def to_numpy(self):
        raise NotImplementedError

#TODO(vertix): Introduce damage into state

class LivingUnitState(State):
    def __init__(self, unit, me, game):
        self.unit = unit
        self.me = me
        self.game = game

    @property
    def hp(self):
        return self.unit.life

    @property
    def max_hp(self):
        return self.unit.max_life

    @property
    def mana(self):
        return 0.

    @property
    def max_mana(self):
        return 0.

    @property
    def position(self):
        return (self.unit.x, self.unit.y)

    @property
    def speed(self):
        return (self.unit.speed_x, self.unit.speed_y)

    @property
    def angle(self):
        return self.unit.angle

    @property
    def rel_position(self):
        return (self.unit.x - self.me.x, self.unit.y - self.me.y)

    @property
    def radius(self):
        return self.unit.radius

    @property
    def rel_speed(self):
        return (self.unit.speed_x - self.me.speed_x, self.unit.speed_y - self.me.speed_x)

    @property
    def max_speed(self):
        """Maximum speed that unit can make"""
        return 0.

    @property
    def rel_angle(self):
        ang = self.unit.angle - self.me.angle
        if ang < 0:
            ang += 2 * math.pi
        if ang > 2 * math.pi:
            ang -= 2 * math.pi
        return ang

    @property
    def dist(self):
        return self.unit.get_distance_to_unit(self.me)

    @property
    def attack_range(self):
        return 0.

    @property
    def vision_range(self):
        return self.unit.vision_range

    @property
    def cooldown_ticks(self):
        """Ticks left to attack"""
        return 0.

    @property
    def enemy(self):
        """Returns 1. if enemy, else 0."""
        return 1. if self.unit.faction == 1 - self.me.faction else 0.

    @property
    def neutral(self):
        """1. if neutral unit, else 0."""
        return 1. if self.unit.faction > Faction.RENEGADES else 0.

    def to_numpy(self):
        return np.array([
            self.hp, self.max_hp, self.mana, self.max_mana,
            self.position[0], self.position[1], self.radius, self.speed[0], self.speed[1],
            self.max_speed, self.angle, self.rel_position[0], self.rel_position[1],
            self.rel_speed[0], self.rel_speed[1], self.rel_angle,
            self.dist, self.attack_range, self.vision_range, self.cooldown_ticks
        ])


class ZeroState(LivingUnitState):
    def __init__(self):
        super(ZeroState, self).__init__(None, None, None)

    @property
    def hp(self):
        return 0.
    @property
    def max_hp(self):
        return 0.
    @property
    def position(self):
        return (0., 0.)
    @property
    def speed(self):
        return (0., 0.)
    @property
    def angle(self):
        return 0.
    @property
    def rel_position(self):
        return (0., 0.)
    @property
    def radius(self):
        return 0.
    @property
    def rel_speed(self):
        return (0., 0.)
    @property
    def max_speed(self):
        return 0.
    @property
    def rel_angle(self):
        return 0.
    @property
    def dist(self):
        return 0.
    @property
    def attack_range(self):
        return 0.
    @property
    def vision_range(self):
        return 0.
    @property
    def cooldown_ticks(self):
        return 0.
    @property
    def enemy(self):
        return 0.
    @property
    def neutral(self):
        return 0.


class WizardState(LivingUnitState):
    def __init__(self, w, me, game):
        super(WizardState, self).__init__(w, me, game)

    @property
    def mana(self):
        return self.unit.mana

    @property
    def max_mana(self):
        return self.unit.max_mana

    @property
    def max_speed(self):
        return 0.0  # TODO(vertix): Encode max speed too.

    @property
    def attack_range(self):
        return self.unit.cast_range

    @property
    def cooldown_ticks(self):
        return self.unit.remaining_action_cooldown_ticks


class BuildingState(LivingUnitState):
    def __init__(self, b, me, game):
        super(BuildingState, self).__init__(b, me, game)

    @property
    def attack_range(self):
        return self.unit.attack_range

    @property
    def cooldown_ticks(self):
        return self.unit.remaining_action_cooldown_ticks


class MinionState(LivingUnitState):
    def __init__(self, m, me, game):
        super(MinionState, self).__init__(m, me, game)

    @property
    def attack_range(self):
        if self.unit.type == MinionType.ORC_WOODCUTTER:
            return self.game.orc_woodcutter_attack_range
        else:
            return self.game.fetish_blowdart_attack_range

    @property
    def cooldown_ticks(self):
        return self.unit.remaining_action_cooldown_ticks


class MyState(WizardState):
    def __init__(self, me, game):
        super(MyState, self).__init__(me, me, game)

    def to_numpy(self):
        return np.array([
            self.hp, self.max_hp, self.mana, self.max_mana,
            self.position[0], self.position[1], self.radius, self.speed[0], self.speed[1],
            self.max_speed, self.angle,
            self.attack_range, self.vision_range, self.cooldown_ticks
        ])


ZERO_NUMPY = ZeroState().to_numpy()
MAX_ENEMIES = 10
MAX_FRIENDS = 10


class WorldState(State):
    def __init__(self, me, world, game):
        super(WorldState, self).__init__()
        self.my_state = MyState(me, game)

        states = [WizardState(w, me, game) for w in world.wizards if w != me]
        states += [MinionState(m, me, game) for m in world.minions
                   if m.faction not in [Faction.NEUTRAL, Faction.OTHER]]
        states += [BuildingState(b, me, game) for b in world.buildings]

        states = sorted(states, key=lambda x: x.dist)

        self.enemy_states = [s for s in states if s.enemy][:MAX_ENEMIES]
        self.friend_states = [s for s in states if not s.enemy][:MAX_FRIENDS]

    def to_numpy(self):
        return np.hstack([self.my_state.to_numpy()] +
                         [s.to_numpy() for s in self.enemy_states] +
                         [ZERO_NUMPY] * (MAX_ENEMIES - len(self.enemy_states)) +
                         [s.to_numpy() for s in self.friend_states] +
                         [ZERO_NUMPY] * (MAX_FRIENDS - len(self.friend_states)))
