import bisect
import collections
import itertools
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
from model.SkillType import SkillType
from model.StatusType import StatusType
from model.ProjectileType import ProjectileType
from model.Projectile import Projectile

from Analysis import GetAggro
from Analysis import IsEnemy
from Analysis import BuildMinionTargets
from Analysis import Closest
from Analysis import NeutralMinionInactive
from Analysis import PickBestFireballTarget
from Analysis import ActionToProjectileTypeAndRadius

from Geometry import GetLanes
from Geometry import Point
from Geometry import Line
from Geometry import Segment
from Geometry import PlainCircle
from Geometry import BuildPaths

from copy import deepcopy

from Colors import BLACK
from Colors import GREEN

MACRO_EPSILON=1.0
GRAPH_REBUILD_FREQUENCY=50


class LUType:
    WIZARD = 0
    MINION = 1
    BUILDING = 2

# Ideas for features
# Exp for hit
# Exp for kill
# Damage
# Av. damage per tick OR Max cooldown
# Regeneration per tick
# Verify that wizard bonuses are used in State
# Received damage per last N ticks.
# Current tick

class State(object):
    """Base class for state"""
    def __init__(self, unit, me, dbg):
        self.numpy_cache = None
        self.dbg = dbg
        self.dist_cache = unit.get_distance_to_unit(me) if unit is not None else 0.0

    @property
    def dist(self):
        return self.dist_cache

    def dbg_text(self, pos, text, color=BLACK):
        if self.dbg is None:
            return
        lines = str(text).split('\n')
        y = pos.y
        for l in lines:
            self.dbg.text(pos.x, y, l, color)
            y += 14

    def dbg_line(self, p1, p2, color=BLACK):
        if self.dbg is None:
            return
        self.dbg.line(p1.x, p1.y, p2.x, p2.y, color)

    def dbg_circle(self, c, color=BLACK):
        if self.dbg is None:
            return
        self.dbg.circle(c.x, c.y, c.radius, color)

    @property
    def enemy(self):
        """Returns 1. if enemy, else 0."""
        return 0.

    def _to_numpy_internal(self):
        raise NotImplementedError

    def to_numpy(self):
        if self.numpy_cache is None:
            self.numpy_cache = self._to_numpy_internal()
        return self.numpy_cache


class TreeState(State):
    def __init__(self, t, me, dbg):
        super(TreeState, self).__init__(t, me, dbg)
        self.unit = t


class ProjectileState(State):
    def __init__(self, p, me, game, world, state, dbg):
        super(ProjectileState, self).__init__(p, me, dbg)
        last_state = state.last_state
        self.p = self.unit = p
        self.dots = 0.0
        self.game = game
        if p.id >= 0 and (p.id in last_state.index):
            old_p = last_state.index[p.id]
            self.min_damage = old_p.min_damage
            self.max_damage = old_p.max_damage
            self.expected_end = old_p.expected_end
            self.expected_speed = old_p.expected_speed
            self.border1 = old_p.border1
            self.border2 = old_p.border2
            self.center_line = old_p.center_line
            return
        need_to_subtract = False
        if p.owner_unit_id in last_state.index:
            owner = last_state.index[p.owner_unit_id]
            need_to_subtract = True
        elif p.owner_unit_id in state.last_seen:
            owner = state.last_seen[p.owner_unit_id]
            need_to_subtract = True
        else:
            self.expected_end = Point.FromUnit(self.unit)
            self.min_damage = self.max_damage = 0.0
            self.expected_speed = math.hypot(p.speed_x, p.speed_y)
            self.border1 = self.border2 = self.center_line = Segment(self.start, self.end)
            return
        if p.id < 0:
            need_to_subtract = False

        if self.p.type == ProjectileType.FIREBALL:
            self.expected_speed = self.game.fireball_speed
            self.min_damage = owner.get_effective_damage_by_me(
                self.game.fireball_explosion_min_damage)
            self.max_damage = owner.get_effective_damage_by_me(
                self.game.fireball_explosion_max_damage)
        if self.p.type == ProjectileType.FROST_BOLT:
            self.expected_speed = self.game.frost_bolt_speed
            self.min_damage = self.max_damage = owner.frost_bolt
        if self.p.type == ProjectileType.MAGIC_MISSILE:
            self.expected_speed = self.game.magic_missile_speed
            self.min_damage = self.max_damage = owner.missile
        if self.p.type == ProjectileType.DART:
            self.expected_speed = self.game.dart_speed
            self.min_damage = self.max_damage = self.game.dart_direct_damage
        speed = Point(p.speed_x, p.speed_y)
        speed *= (1.0 / speed.Norm())
        cast_range = game.fetish_blowdart_attack_range
        if self.p.type != ProjectileType.DART:
            cast_range = owner.unit.cast_range
        start_point = Point.FromUnit(p)
        if need_to_subtract:
            start_point -= (speed * self.expected_speed)
        self.expected_end = start_point + speed * cast_range
        for t in world.trees:
            tp = Point.FromUnit(t)
            ray = Segment(self.start, self.expected_end)
            d = tp.GetSqDistanceToSegment(ray)
            if d < (t.radius + self.min_radius) * (t.radius + self.min_radius) - MACRO_EPSILON:
                pc = PlainCircle(tp, t.radius + self.min_radius)
                # intersect segment ray with pc, pick closest to start
                intersections = ray.l.IntersectWithCircle(pc)
                dists = [i.GetSqDistanceTo(self.start) for i in intersections]
                if dists[0] < dists[1]:
                    self.expected_end = intersections[0]
                else:
                    self.expected_end = intersections[1]
                self.dbg_line(start_point, self.expected_end, GREEN)
        self.center_line = Segment(self.expected_end, start_point)
        shift = self.center_line.l.Normal() * (self.max_radius + me.radius + MACRO_EPSILON)
        self.border1 = Segment(
            start_point + shift,
            self.expected_end + shift)
        self.border2 = Segment(
            start_point - shift,
            self.expected_end - shift)

    @property
    def start(self):
        return Point.FromUnit(self.unit)

    @property
    def end(self):
        return self.expected_end

    @property
    def min_radius(self):
        if self.p.type == ProjectileType.FIREBALL:
            return max(self.p.radius, self.game.fireball_explosion_max_damage_range)
        
        return self.p.radius

    @property
    def max_radius(self):
        if self.p.type == ProjectileType.FIREBALL:
            return self.game.fireball_explosion_min_damage_range
        
        return self.p.radius
        
    @property
    def min_direct_damage(self):
        return self.min_damage

    @property
    def max_direct_damage(self):
        return self.max_damage

    @property
    def overtime_damage(self):
        return (self.game.burning_summary_damage 
                if self.p.type == ProjectileType.FIREBALL
                else 0.0)
    
    @property
    def freeze_time(self):
        return (self.game.frozen_duration_ticks 
                if self.p.type == ProjectileType.FROST_BOLT
                else 0.0)
    
    @property
    def speed(self):
        return self.expected_speed

    def _to_numpy_internal(self):
        return np.array([
            self.start.x / 1000., self.start.y / 1000.,
            self.end.x  / 1000., self.end.y  / 1000.,
            self.min_radius, self.max_radius, self.min_direct_damage / 10.,
            self.max_direct_damage / 10., self.overtime_damage / 10.,
            self.speed / 10., self.freeze_time  / 10.
        ])

PROJECTILE_STATE_SIZE = 11

def combine_damage_caused(damage_caused, game):
    return ((1 + game.wizard_damage_score_factor) * damage_caused[LUType.WIZARD] +
            (1 + game.building_damage_score_factor) * damage_caused[LUType.BUILDING] + 
            (1 + game.minion_damage_score_factor) * damage_caused[LUType.MINION])

class LivingUnitState(State):
    def __init__(self, unit, me, game, world, dbg):
        super(LivingUnitState, self).__init__(unit, me, dbg)
        self.unit = unit
        self.me = me
        self.game = game
        self.world = world
        self.cache_rel_angle = me.get_angle_to_unit(unit) if me is not None else 0.0
        self.preds = []
        self.dots = 0.0
        if unit is None:
            return
        for s in unit.statuses:
            if s.type == StatusType.BURNING:
                self.dots += (1.0 * s.remaining_duration_ticks /
                              game.burning_duration_ticks *
                              game.burning_summary_damage)

    def get_effective_damage_to_me(self, nominal_damage):
        return nominal_damage

    @property
    def type(self):
        return -1

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
        return self.cache_rel_angle

    @property
    def attack_range(self):
        return 0.

    @property
    def aggro_range(self):
        return 0.

    @property
    def vision_range(self):
        return self.unit.vision_range

    @property
    def cooldown_ticks(self):
        """Ticks left to attack"""
        return self.unit.remaining_action_cooldown_ticks

    @property
    def total_cooldown_ticks(self):
        """Ticks between attacks"""
        return self.unit.cooldown_ticks

    @property
    def enemy(self):
        """Returns 1. if enemy, else 0."""
        return 1. if IsEnemy(self.me, self.unit) else 0.

    @property
    def neutral(self):
        """1. if neutral unit, else 0."""
        return 1. if self.unit.faction > Faction.RENEGADES else 0.

    @property
    def is_on_top_lane(self):
        return 1. if LaneType.TOP in GetLanes(self.unit) else 0.

    @property
    def is_on_middle_lane(self):
        return 1. if LaneType.MIDDLE in GetLanes(self.unit) else 0.

    @property
    def is_on_bottom_lane(self):
        return 1. if LaneType.BOTTOM in GetLanes(self.unit) else 0.

    @property
    def expected_overtime_damage(self):
        return self.dots

    @property
    def frost_bolt(self):
        return 0.0

    @property
    def frost_bolt_cooldown(self):
        return 0.0

    @property
    def fireball(self):
        return 0.0

    @property
    def fireball_cooldown(self):
        return 0.0

    @property
    def missile(self):
        return 0.0

    @property
    def missile_cooldown(self):
        return 0.0

    @property
    def staff(self):
        return 0.0

    @property
    def hp_regen(self):
        return 0.0

    @property
    def forward_speed(self):
        return 0.0

    @property
    def strafe_speed(self):
        return 0.0

    @property
    def damage(self):
        return self.unit.damage

    @property
    def damage_per_tick(self):
        return float(self.damage) / self.total_cooldown_ticks

    @property
    def predictions(self):
        return self.preds

    def projected_living_time(self, incoming_dpt=0.0):
        idx = self.projected_death_pred_idx(incoming_dpt)
        if idx >= len(self.preds):
            return 20000.
        return self.preds[idx].time
        
    def projected_damage_prevented(self, incoming_dpt=0.0):
        idx = self.projected_death_pred_idx(incoming_dpt)
        if idx >= len(self.preds):
            return 0.
        return combine_damage_caused(
            self.preds[len(self.preds) - 1].damage_caused, self.game) - combine_damage_caused(
                self.preds[idx].damage_caused, self.game)
    
    def projected_death_pred_idx(self, incoming_dpt=0.0):
        left, right = 0, len(self.preds)
        while left < right:
            mid = (left + right) / 2
            if self.preds[mid].hp - self.preds[mid].time * incoming_dpt <= 0:
                right = mid
            else:
                left = mid + 1
        return left
        

    def SetPredictions(self, preds):
        self.preds = preds

    def _to_numpy_internal(self):
        return np.array([
            self.hp / 100., self.max_hp / 100., self.mana / 100., self.max_mana / 100.,
            self.position[0] / 1000., self.position[1] / 1000.,
            self.radius / 10., self.speed[0], self.speed[1],
            self.max_speed, self.angle,
            self.rel_position[0] / 1000., self.rel_position[1] / 1000.,
            self.rel_speed[0], self.rel_speed[1], self.rel_angle,
            self.dist / 1000., self.attack_range / 1000., self.vision_range / 1000.,
            self.cooldown_ticks / 100.,  # 19
            self.is_on_top_lane, self.is_on_middle_lane, self.is_on_bottom_lane,
            self.total_cooldown_ticks / 100., self.aggro_range / 100.,
            self.expected_overtime_damage / 10.,
            self.frost_bolt / 10., self.frost_bolt_cooldown / 10.,
            self.fireball / 10., self.fireball_cooldown / 10.,
            self.missile / 10., self.missile_cooldown / 10.,
            self.staff, self.hp_regen, self.forward_speed, self.strafe_speed
        ])


LIVING_UNIT_STATE_SIZE = 36


class WizardState(LivingUnitState):
    def __init__(self, w, me, game, world, dbg):
        super(WizardState, self).__init__(w, me, game, world, dbg)
        missile_radius = game.magic_missile_radius
        self.has_frost_bolt = SkillType.FROST_BOLT in w.skills
        self.has_fireball = SkillType.FIREBALL in w.skills
        missile_radius = game.magic_missile_radius
        if self.has_frost_bolt:
            missile_radius = max(missile_radius, game.frost_bolt_radius)
        if self.has_fireball:
            missile_radius = max(missile_radius, game.fireball_explosion_min_damage_range)
        # me.radius is needed to make towers work uniformly
        self.effective_range = w.cast_range + missile_radius + me.radius
        self.hastened = StatusType.HASTENED in [st.type for st in w.statuses]

        self.handle_improvements(w, game, world)

        self.cached_missile_total_cooldown = game.magic_missile_cooldown_ticks
        if SkillType.ADVANCED_MAGIC_MISSILE in self.unit.skills:
            self.cached_missile_total_cooldown = game.wizard_action_cooldown_ticks
        if w.faction != me.faction and self.fireball and self.fireball_cooldown == 0:
            self.add_fake_projectile(ActionType.FIREBALL, me, world, game)
        if w.faction != me.faction and self.frost_bolt and self.frost_bolt_cooldown == 0:
            self.add_fake_projectile(ActionType.FROST_BOLT, me, world, game)
        if w.faction != me.faction and self.missile_cooldown == 0:
            self.add_fake_projectile(ActionType.MAGIC_MISSILE, me, world, game)

    def add_fake_projectile(self, action, me, world, game):
        da = self.unit.get_angle_to_unit(me)
        speed = Point.FromUnit(self.unit) - Point.FromUnit(me)
        if abs(da) > self.max_rotation_speed:
            da = da / abs(da) * self.max_rotation_speed
            speed = Point(1.0, 0.0).Rotate(self.unit.angle + da)
        t, r = ActionToProjectileTypeAndRadius(action, game)
        fake_p = Projectile(id=(self.unit.id + 1000) * (-5) - action,
            x=self.unit.x, y=self.unit.y,
            speed_x=speed.x, speed_y=speed.y,
            angle=speed.GetAngle(), faction=self.unit.faction,
            radius=r, type=t, owner_unit_id=self.unit.id,
            owner_player_id=self.unit.owner_player_id)
        self.world.projectiles.append(fake_p)

    def handle_improvements(self, w, game, world):
        haste_aura1 = False
        haste_aura2 = False
        m_damage_aura1 = False
        m_damage_aura2 = False
        s_damage_aura1 = False
        s_damage_aura2 = False
        shield_aura1 = False
        shield_aura2 = False
        for ww in world.wizards:
            if ((ww.faction == w.faction) and 
                (Point.FromUnit(ww).GetSqDistanceTo(w) < 
                 game.aura_skill_range * game.aura_skill_range)): 
                if SkillType.MOVEMENT_BONUS_FACTOR_AURA_1 in ww.skills:
                    haste_aura1 = True
                if SkillType.MOVEMENT_BONUS_FACTOR_AURA_2 in ww.skills:
                    haste_aura2 = True
                if SkillType.MAGICAL_DAMAGE_BONUS_AURA_1 in ww.skills:
                    m_damage_aura1 = True
                if SkillType.MAGICAL_DAMAGE_BONUS_AURA_2 in ww.skills:
                    m_damage_aura2 = True
                if SkillType.STAFF_DAMAGE_BONUS_AURA_1 in ww.skills:
                    s_damage_aura1 = True
                if SkillType.STAFF_DAMAGE_BONUS_AURA_2 in ww.skills:
                    s_damage_aura2 = True
                if SkillType.MAGICAL_DAMAGE_ABSORPTION_AURA_1 in ww.skills:
                    shield_aura1 = True
                if SkillType.MAGICAL_DAMAGE_ABSORPTION_AURA_2 in ww.skills:
                    shield_aura2 = True
        haste_factor = 1.0
        if self.hastened:
            haste_factor += game.hastened_movement_bonus_factor
        if SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_1 in w.skills:
            haste_factor += game.movement_bonus_factor_per_skill_level
        if SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_2 in w.skills:
            haste_factor += game.movement_bonus_factor_per_skill_level
        if haste_aura1:
            haste_factor += game.movement_bonus_factor_per_skill_level
        if haste_aura2:
            haste_factor += game.movement_bonus_factor_per_skill_level
        self.max_strafe_speed = game.wizard_strafe_speed * haste_factor
        self.max_forward_speed = game.wizard_forward_speed * haste_factor
        self.max_effective_speed = self.max_forward_speed
        
        self.damage_factor = 1.0
        if StatusType.EMPOWERED in [st.type for st in w.statuses]:
            self.damage_factor += game.empowered_damage_factor
        self.m_damage_increase = 0.0
        if SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_1 in w.skills:
            self.m_damage_increase += game.magical_damage_bonus_per_skill_level
        if SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_2 in w.skills:
            self.m_damage_increase += game.magical_damage_bonus_per_skill_level
        if m_damage_aura1:
            self.m_damage_increase += game.magical_damage_bonus_per_skill_level
        if m_damage_aura2:
            self.m_damage_increase += game.magical_damage_bonus_per_skill_level
        self.s_damage_increase = 0.0
        if SkillType.STAFF_DAMAGE_BONUS_PASSIVE_1 in w.skills:
            self.s_damage_increase += game.staff_damage_bonus_per_skill_level
        if SkillType.STAFF_DAMAGE_BONUS_PASSIVE_2 in w.skills:
            self.s_damage_increase += game.staff_damage_bonus_per_skill_level
        if s_damage_aura1:
            self.s_damage_increase += game.staff_damage_bonus_per_skill_level
        if s_damage_aura2:
            self.s_damage_increase += game.staff_damage_bonus_per_skill_level

        self.absorption = 0.0
        self.absorption_factor = 0.0
        if StatusType.SHIELDED in [st.type for st in w.statuses]:
            self.absorption_factor += game.shielded_direct_damage_absorption_factor
        if SkillType.MAGICAL_DAMAGE_ABSORPTION_PASSIVE_1 in w.skills:
            self.absorption += game.magical_damage_absorption_per_skill_level
        if SkillType.MAGICAL_DAMAGE_ABSORPTION_PASSIVE_2 in w.skills:
            self.absorption += game.magical_damage_absorption_per_skill_level
        if shield_aura1:
            self.absorption += game.magical_damage_absorption_per_skill_level
        if shield_aura2:
            self.absorption += game.magical_damage_absorption_per_skill_level

    def get_effective_damage_to_me(self, nominal_damage):
        return max(0.0, nominal_damage * (1.0 - self.absorption_factor) - self.absorption)

    def get_effective_damage_by_me(self, nominal_damage):
        return (nominal_damage + self.m_damage_increase) * self.damage_factor

    def remaining_action_cooldown(self, action):
        if action == ActionType.MAGIC_MISSILE:
            return self.missile_cooldown
        if action == ActionType.FIREBALL:
            return self.fireball_cooldown
        if action == ActionType.FROST_BOLT:
            return self.frost_bolt_cooldown
        return self.staff_cooldown

    @property
    def type(self):
        return LUType.WIZARD

    @property
    def mana(self):
        return self.unit.mana

    @property
    def max_mana(self):
        return self.unit.max_mana

    @property
    def max_speed(self):
        return self.max_effective_speed

    @property
    def max_rotation_speed(self):
        speed = self.game.wizard_max_turn_angle
        if StatusType.HASTENED in [st.type for st in self.unit.statuses]:
            speed *= (1.0 + self.game.hastened_rotation_bonus_factor)
        return speed

    @property
    def attack_range(self):
        return self.effective_range

    @property
    def aggro_range(self):
        return self.effective_range

    @property
    def cooldown_ticks(self):
        return max(self.unit.remaining_action_cooldown_ticks,
                   min(self.missile_cooldown, self.fireball_cooldown,
                       self.frost_bolt_cooldown))

    @property
    def damage(self):
        return max(self.fireball, self.frost_bolt, self.missile)

    @property
    def staff_cooldown(self):
        return self.unit.remaining_action_cooldown_ticks

    @property
    def frost_bolt_total_cooldown(self):
        return self.game.frost_bolt_cooldown_ticks

    @property
    def fireball_total_cooldown(self):
        return self.game.fireball_cooldown_ticks
        
    @property
    def missile_total_cooldown(self):
        return self.cached_missile_total_cooldown

    @property
    def total_cooldown_ticks(self):
        """Ticks between attacks"""
        return max(self.game.wizard_action_cooldown_ticks, 
                   self.missile_total_cooldown)

    @property
    def cooldown_ticks(self):
        """Ticks until next attack"""
        min_action_cooldown = self.missile_cooldown
        if self.has_fireball:
            min_action_cooldown = min(min_action_cooldown, self.fireball_cooldown)
        if self.has_frost_bolt:
            min_action_cooldown = min(min_action_cooldown, self.frost_bolt_cooldown)
        return max(self.unit.remaining_action_cooldown_ticks, min_action_cooldown)
    
    @property
    def mana_regen(self):
        return (self.game.wizard_base_mana_regeneration +
                self.unit.level * self.game.wizard_mana_regeneration_growth_per_level)
    
    @property
    def frost_bolt(self):
        return (self.get_effective_damage_by_me(self.game.frost_bolt_direct_damage) 
                if self.has_frost_bolt else 0.0)

    @property
    def frost_bolt_cooldown(self):
        return max(self.unit.remaining_cooldown_ticks_by_action[ActionType.FROST_BOLT],
                   (self.game.frost_bolt_manacost - self.mana) / self.mana_regen,
                   self.unit.remaining_action_cooldown_ticks)

    @property
    def fireball(self):
        return (self.fireball_direct_damage + self.game.burning_summary_damage
                if self.has_fireball else 0.0)
        
    @property
    def fireball_direct_damage(self):
        return self.get_effective_damage_by_me(self.game.fireball_explosion_max_damage)
        
    @property
    def fireball_cooldown(self):
        return max(self.unit.remaining_cooldown_ticks_by_action[ActionType.FIREBALL],
                   (self.game.fireball_manacost - self.mana) / self.mana_regen,
                   self.unit.remaining_action_cooldown_ticks)
        
    @property
    def missile(self):
        return self.get_effective_damage_by_me(self.game.magic_missile_direct_damage)
        
    @property
    def missile_cooldown(self):
        return max(self.unit.remaining_action_cooldown_ticks,
                   self.unit.remaining_cooldown_ticks_by_action[ActionType.MAGIC_MISSILE],
                   (self.game.magic_missile_manacost - self.mana) / self.mana_regen)
    
    @property
    def staff(self):
        return (self.game.staff_damage + self.s_damage_increase) * self.damage_factor

    @property
    def hp_regen(self):
        return (self.game.wizard_base_life_regeneration +
                self.unit.level * self.game.wizard_life_regeneration_growth_per_level)

    @property
    def forward_speed(self):
        return self.max_forward_speed

    @property
    def strafe_speed(self):
        return self.max_strafe_speed


class BuildingState(LivingUnitState):
    def __init__(self, b, me, game, world, dbg):
        super(BuildingState, self).__init__(b, me, game, world, dbg)

    @property
    def type(self):
        return LUType.BUILDING

    @property
    def attack_range(self):
        return self.unit.attack_range

    @property
    def aggro_range(self):
        return self.unit.attack_range

    @property
    def cooldown_ticks(self):
        return self.unit.remaining_action_cooldown_ticks

    @property
    def total_cooldown_ticks(self):
        return self.unit.cooldown_ticks


class MinionState(LivingUnitState):
    def __init__(self, m, me, game, world, global_state):
        super(MinionState, self).__init__(m, me, game, world, global_state.dbg)
        self.cached_attack_range = (
            self.game.fetish_blowdart_attack_range + self.game.dart_radius + me.radius)
        if self.unit.type == MinionType.ORC_WOODCUTTER:
            self.cached_attack_range = self.game.orc_woodcutter_attack_range + me.radius
        if (m.faction == Faction.NEUTRAL) and NeutralMinionInactive(m):
            self.cached_aggro_range = 0.0
        else:
            self.cached_aggro_range = min(
                self.cached_attack_range,
                Closest(m, global_state.minion_targets[m.faction]))

    @property
    def type(self):
        return LUType.MINION

    @property
    def attack_range(self):
        return self.cached_attack_range

    @property
    def aggro_range(self):
        return self.cached_aggro_range
        
    @property
    def max_speed(self):
        return self.game.minion_speed

    @property
    def cooldown_ticks(self):
        return self.unit.remaining_action_cooldown_ticks

    @property
    def total_cooldown_ticks(self):
        return self.unit.cooldown_ticks
        
    @property
    def forward_speed(self):
        return self.game.minion_speed



class MyState(WizardState):
    def __init__(self, me, game, world, lane, world_state):
        super(MyState, self).__init__(me, me, game, world, world_state.dbg)
        self.lane = lane
        world_state.index[me.id] = self
        self.cached_aggro = GetAggro(me, 10, world_state)
        self.fireball_target = None
        self.fireball_projected_damage = 0
        if self.fireball_cooldown < 3:
            target_and_damage = PickBestFireballTarget(me, world_state)
            if target_and_damage is not None:
                self.fireball_projected_damage = target_and_damage.CombinedDamage(world_state)
                self.fireball_target = target_and_damage.target
        
    @property
    def max_fireball_damage(self):
        return self.fireball_projected_damage

    @property
    def aggro(self):
        return self.cached_aggro

    @property
    def current_lane(self):
        return self.lane

    def _to_numpy_internal(self):
        return np.array([
            self.hp / 100., self.max_hp / 100., self.mana / 100., self.max_mana / 100.,
            self.position[0] / 1000., self.position[1] / 1000.,
            self.speed[0], self.speed[1],
            self.max_speed, self.angle,
            self.attack_range / 1000., self.vision_range / 1000.,
            self.cooldown_ticks / 100.,
            self.is_on_top_lane, self.is_on_middle_lane, self.is_on_bottom_lane,
            self.total_cooldown_ticks / 100., self.aggro_range / 100.,
            self.expected_overtime_damage / 10.,
            self.frost_bolt / 10., self.frost_bolt_cooldown / 10.,
            self.fireball / 10., self.fireball_cooldown / 10.,
            self.missile / 10., self.missile_cooldown / 10.,
            self.staff, self.hp_regen, self.forward_speed, self.strafse_speed,
            self.aggro / 10., self.current_lane, self.max_fireball_damage
        ])  # Size is 32

MAX_ENEMIES = 20
MAX_FRIENDS = 20
MAX_PROJECTILES = 5

def clean_world(me, world):
    new_trees = []
    for t in world.trees:
        if t.get_distance_to_unit(me) < 1000:
            new_trees.append(t)
    world.trees = new_trees
    new_p = []
    for p in world.projectiles:
        if p.get_distance_to_unit(me) < 1000:
            new_p.append(p)
    world.projectiles = new_p


class UnitPrediction(collections.namedtuple(
    'UnitPrediction', ['time', 'hp', 'pos', 'damage_caused', 'deaths_caused'])):

    def __str__(self):
        return '%d -> hp%d %s %s' % (self.time, self.hp, 
                                     ['%d: %.1f' % i for i in self.damage_caused.items()],
                                     ['%d: %.1f' % i for i in self.deaths_caused.items()])


class UnitModel(object):
    def __init__(self, unit_state):
        self.unit_state = unit_state
        self.position = Point.FromUnit(unit_state.unit)
        self.hp = unit_state.hp
        self.dpt = 0.
        self.speed = 0.
        if unit_state.enemy or unit_state.type != LUType.WIZARD:
            self.dpt = unit_state.damage_per_tick
            self.speed = unit_state.max_speed

        self.start_hp = unit_state.hp
        self.damage_caused = collections.defaultdict(float)
        self.deaths_caused = collections.defaultdict(float)

        self.target = None
        self.target_position = None
        # Maps attacker into damage inflicted
        self.attackers = collections.defaultdict(float)
        self.current_attackers = set()
        self.next_attackers = set()

    def __str__(self):
        return '[%s%s, hp=%.1f, (%d, %d)]' % (
            ['Wz', 'Mn', 'Bd'][self.unit_state.type],
            '+-'[int(self.unit_state.enemy)],
            self.hp,
            self.position.x,
            self.position.y,
            # self.unit_state.rel_position[0],
            # self.unit_state.rel_position[1]
            )

    def __repr__(self):
        return str(self)

    @property
    def is_corpse(self):
        return self.hp <= 0.

    def Distance(self, model):
        return model.position.GetDistanceTo(self.position)

    @property
    def distance_to_target(self):
        return self.position.GetDistanceTo(self.target.position)

    def SetEnemies(self, enemies):
        self.enemies = enemies

    def SelectTarget(self):
        # TODO(vertix): Implement right logic. Now we just select the closest.
        if not self.enemies:
            self.target = None
            return

        if self.unit_state.type == LUType.MINION:
            new_target = min(self.enemies, key=lambda u: self.Distance(u))
        else:
            def SmartDamageKey(target):
                if self.Distance(target) > self.unit_state.attack_range:
                    return 100500. + self.Distance(target)
                if target.hp > self.unit_state.damage:
                    return target.hp
                return -target.hp
            new_target = min(self.enemies, key=SmartDamageKey)

        if new_target != self.target and self.target and self in self.target.current_attackers:
            self.target.current_attackers.remove(self)

        self.target = new_target

        if self.Distance(self.target) <= self.unit_state.attack_range:
            # print '  %s attacks %s' % (self, self.target)
            self.target.current_attackers.add(self)
            # self.target.next_attackers.add(self)

    def ModelTimePeriod(self, time):
        # TODO(vertix): We should model the movement of attackers when I move!
        if self.distance_to_target > self.unit_state.attack_range:
            vec = self.target.position - self.position
            self.position += vec * (time * self.speed / vec.Norm())
            if self.distance_to_target < self.unit_state.attack_range:
                self.target.next_attackers.add(self)   # Now we are ready to attack.

        # Now we model attack on this unit.
        for a in self.current_attackers:
            # print "# %s damages %s for %d" % (a, self, time)

            dmg = time * a.dpt
            # print '-- Inferred damage %.1f' % dmg
            a.damage_caused[self.unit_state.type] += dmg
            self.attackers[a] += dmg
            self.hp -= dmg
        self.hp = max(0., min(self.unit_state.max_hp, self.hp + time * self.unit_state.hp_regen))

        if self.hp <= 0:
            tot_dmg = sum(self.attackers.itervalues())
            # Record my death to my attackers
            for a, dmg in self.attackers.iteritems():
                a.deaths_caused[self.unit_state.type] += dmg / tot_dmg

    def UpdateAttackers(self):
        if self.hp <= 0 and self in self.target.current_attackers:
            # Stop attacking
            self.target.current_attackers.remove(self)
        return

    def NextEvent(self):
        """Ticks until the next interesting event"""
        dist_to_attack = self.Distance(self.target) - self.unit_state.attack_range
        if dist_to_attack > 0 and self.speed > 0:
            move_time = int(math.ceil(dist_to_attack / self.speed))
            if move_time < self.living_time:
                # print '  %s comes to %s in %d' % (self, self.target, move_time)
                return move_time
            # elif self.living_time < 20000:
                # print '  %s dies in %d' % (self, self.living_time)
        return self.living_time

    @property
    def living_time(self):
        """How much time left before the death :)"""
        # TODO(vertix): This can be optmized by caching this sum
        dpt = sum([a.dpt for a in self.current_attackers]) - self.unit_state.hp_regen
        if dpt > 0:
            return int(math.ceil(self.hp / dpt))
        else:
            return 20000.

def ModelEngagement(friends, enemies):
    if not enemies:
        return

    friends = set(UnitModel(u) for u in friends)
    enemies = set(UnitModel(u) for u in enemies)

    events = collections.defaultdict(list)
    ts = 0

    for f in friends:
        f.SetEnemies(enemies)

    for e in enemies:
        e.SetEnemies(friends)

    corpses_storage = []

    while len(friends) > 0 and len(enemies) > 0:
        for u in itertools.chain(friends, enemies):
            u.SelectTarget()

        # TODO(vertix): Use min-heap here?
        u, dt = min(((u, u.NextEvent()) for u in itertools.chain(friends, enemies)),
                    key=lambda (_, b): b)
        if dt > 15000:  # Too long
            break

        # print '->%s : %d' % (u, dt)

        for u in itertools.chain(friends, enemies):
            u.ModelTimePeriod(dt)
        for u in itertools.chain(friends, enemies):
            u.UpdateAttackers()
        ts += dt

        corpses = []
        for u in itertools.chain(friends, enemies):
            events[u.unit_state].append(
                UnitPrediction(ts, u.hp, u.position,
                               u.damage_caused.copy(),
                               u.deaths_caused.copy()))
            if u.is_corpse:
                corpses.append(u)

        for c in corpses:
            # for a in corpse.attackers:
            #     a.SelectTarget()
            if c in friends: friends.remove(c)
            if c in enemies: enemies.remove(c)

        corpses_storage.extend(corpses)

    for c in corpses_storage:
        if events[c.unit_state][-1].time != ts:
            events[c.unit_state].append(
                UnitPrediction(ts, c.hp, c.position,
                               c.damage_caused.copy(),
                               c.deaths_caused.copy()))

    return events


class WorldState(State):
    def __init__(self, me, world, game, lane, last_state, dbg):
        super(WorldState, self).__init__(None, me, dbg)
        self.my_state = None
        self.events = []
        if last_state is not None:
            last_state.last_state = None
            self.last_state = last_state
            self.enemy_base_hp = self.last_state.enemy_base_hp
            self.last_seen = last_state.last_seen
            last_state.last_seen = None
            self.last_flee_target = last_state.last_flee_target
            self.last_advance_target = last_state.last_advance_target
            self.cached_my_state = last_state.my_state
            self.graph = last_state.graph
            self.eng = last_state.eng
            self.wizard_ids = last_state.wizard_ids
        else:
            self.enemy_base_hp = 1000
            self.last_seen = {}
            self.last_flee_target = None
            self.last_advance_target = None
            self.cached_my_state = None
            self.graph = None
            self.eng = None
            self.wizard_ids = sorted([w.id for w in world.wizards if 
                                      w.faction == me.faction and w.id != me.id])

        self.world = world
        clean_world(me, world)
        self.game = game
        self.minion_targets = [list() for _ in xrange(3)]
        for f in [Faction.ACADEMY, Faction.RENEGADES, Faction.NEUTRAL]:
            self.minion_targets[f] = BuildMinionTargets(f, world)

        self.index = {}
        self.tree_states = [TreeState(t, me, dbg) for t in world.trees]
        states = []
        for w in world.wizards:
            if w.id != me.id:
                ws = WizardState(w, me, game, world, dbg)
                states.append(ws)
                self.index[w.id] = ws
        self.projectile_states = [ProjectileState(p, me, game, world, self, dbg)
                                  for p in world.projectiles]

        for s in ([MinionState(m, me, game, world, self) for m in world.minions] +
                  [BuildingState(b, me, game, world, dbg) for b in world.buildings]):
            states.append(s)
            self.index[s.unit.id] = s

        for b in world.buildings:
            if b.type == BuildingType.FACTION_BASE and b.faction != me.faction:
                self.enemy_base_hp = b.life

        states = sorted(states, key=lambda x: x.dist)
        for s in itertools.chain(self.tree_states, self.projectile_states):
            self.index[s.unit.id] = s
        for w in world.wizards:
            if w.id != me.id:
                self.last_seen[w.id] = self.index[w.id]

        self.my_state = MyState(me, game, world, lane, self)

        self.enemy_states = [s for s in states if s.enemy and s.dist < 1000][:MAX_ENEMIES]
        self.friend_states = [s for s in states if not s.enemy and s.dist < 1000][:MAX_FRIENDS]
        
        if ((self.graph is None or 
             (world.tick_index % GRAPH_REBUILD_FREQUENCY == 0)) and 
            self.last_advance_target is not None):
            self.graph = BuildPaths(self, [self.last_advance_target,
                                           self.last_flee_target])

        if (len(self.enemy_states) > 1 and 
           (self.eng is None or (world.tick_index % GRAPH_REBUILD_FREQUENCY == 0))):
            self.eng = self.ModelEngagement()
        if self.eng:
            # Rewrite to bin search
            for k, v in self.eng.iteritems():
                k.SetPredictions(v)
                damage_caused = (0. if len(k.preds) == 0 else 
                                                 k.preds[len(k.preds) - 1].damage_caused)
                # print damage_caused
                k.dbg_text(Point.FromUnit(k.unit) - Point(50, -50),
                           'TTL:%d\nDMG:%.0f' % (k.projected_living_time(),
                                                 combine_damage_caused(damage_caused, game)))

    def ModelEngagement(self):
        res = ModelEngagement(
            [f for f in self.friend_states + [self.my_state]
             if f.unit.get_distance_to_unit(f.me) < 1500.],
            [e for e in self.enemy_states
             if e.unit.get_distance_to_unit(e.me) < 1500.])
        # if res:
        #     for k, v in res.iteritems():
        #         print '<<<<<<<<<   %s   >>>>>>>>>' % (UnitModel(k))
        #         for item in v:
        #             print '  ' + str(item)
        #         print ""
        #     print '---------------------------'
        return res

    def GetMyState(self):
        if self.my_state is not None:
            return self.my_state
        return self.cached_my_state

    @property
    def ticks_until_end(self):
        return self.world.tick_count - self.world.tick_index

    def _to_numpy_internal(self):
        return np.hstack([self.my_state.to_numpy()] +
                         [s.to_numpy() for s in self.enemy_states] +
                         [np.zeros(LIVING_UNIT_STATE_SIZE)] * (MAX_ENEMIES - len(self.enemy_states)) +
                         [s.to_numpy() for s in self.friend_states] +
                         [np.zeros(LIVING_UNIT_STATE_SIZE)] * (MAX_FRIENDS - len(self.friend_states)) +
                         [s.to_numpy() for s in self.projectile_states[:MAX_PROJECTILES]] +
                         [np.zeros(PROJECTILE_STATE_SIZE)] * (MAX_PROJECTILES - len(self.projectile_states)) +
                         [np.array([self.ticks_until_end  / 10000., self.enemy_base_hp / 1000.])])
