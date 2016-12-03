from collections import namedtuple
from copy import deepcopy

from model.Faction import Faction
from model.MinionType import MinionType
from model.Building import Building
from model.Minion import Minion
from model.Wizard import Wizard
from model.ActionType import ActionType
from model.StatusType import StatusType
from model.BuildingType import BuildingType
from model.Projectile import Projectile
from model.ProjectileType import ProjectileType
from model.LivingUnit import LivingUnit

from Geometry import RangeAllowance
from Geometry import GetLanes
from Geometry import Point
from Geometry import PlainCircle
from math import sqrt

from Colors import RED

CAST_RANGE_ERROR = 0
EPSILON = 1e-4
INFINITY = 1e6
AGGRO_TICKS = 20

def GetAggroFromDamage(damage, remaining_cooldown, cooldown, deepness):
    return (max(0, int(AGGRO_TICKS + deepness - remaining_cooldown + cooldown - 1)) /
            int(cooldown) * damage)

def GetUnitAggro(mes, us, deep_in_range, state):
    aggro = 0.0
    speed = mes.forward_speed
    from State import WizardState
    if isinstance(us, WizardState):
        aggro = GetAggroFromDamage(us.fireball,
                         us.fireball_cooldown,
                         us.fireball_total_cooldown,
                         deep_in_range / max(1.0, speed - us.strafe_speed / 3.0))
        aggro += GetAggroFromDamage(us.frost_bolt,
                         us.frost_bolt_cooldown,
                         us.frost_bolt_total_cooldown,
                         deep_in_range / max(1.0, speed - us.strafe_speed / 3.0))
        aggro += GetAggroFromDamage(us.missile,
                         us.missile_cooldown,
                         us.missile_total_cooldown,
                         deep_in_range / max(1.0, speed - us.strafe_speed / 3.0))
        if us.dist < state.game.staff_range:
            aggro += GetAggroFromDamage(us.staff,
                                        us.staff_cooldown,
                                        state.game.staff_cooldown_ticks)
    else:
        aggro = GetAggroFromDamage(us.damage, us.cooldown_ticks, 
                                   us.total_cooldown_ticks,
                                   deep_in_range / max(1.0, (speed - us.max_speed)))
    # state.dbg_text(us.unit, 'a: %.1f' % aggro)
    return aggro

def BuildMinionTargets(faction, world):
    ret = []
    for t in world.buildings + world.wizards + world.minions:
        if t.faction != faction:
            ret.append(t)
    return ret

def Closest(x, units):
    res = INFINITY
    xp = Point.FromUnit(x)
    for u in units:
        d = xp.GetSqDistanceTo(u)
        res = min(res, d)
    return sqrt(max(0.0, res))
    
def ClosestUnit(x, units):
    res = INFINITY
    a = None
    xp = Point.FromUnit(x)
    for u in units:
        d = xp.GetSqDistanceTo(u)
        if res > d:
            res = d
            a = u 
    return a
    
def GetClosestTarget(me, state, radius=INFINITY):
    return ClosestUnit(me, BuildTargets(me, min(radius, me.vision_range*2), state))
    
def BuildEnemies(me, radius, state):
    return [e for e in state.world.wizards + state.world.minions + state.world.buildings if
                   IsEnemy(me, e) 
                   and (e.get_distance_to_unit(me) < radius)]

def BuildTargets(me, radius, state):
    return [t for t in BuildEnemies(me, radius, state) if IsValidTarget(me, t, state)]

def CanDodge(w, ps, state):
    debug_string = ''
    psp = Point.FromUnit(ps.p)
    wp = Point.FromUnit(w)
    d = psp.GetDistanceTo(wp) - w.radius - ps.max_radius
    state.dbg_circle(PlainCircle(ps.end, ps.max_radius + w.radius))
    state.dbg_line(ps.border1.p1, ps.border1.p2)
    state.dbg_line(ps.border2.p1, ps.border2.p2)
    time = d / ps.speed
    distance_to_circle = INFINITY
    # debug_string += 'w_to_start: %.0f\n' % wp.GetDistanceTo(ps.start)
    # debug_string += 'start_to_end: %.0f\n' % ps.start.GetDistanceTo(ps.end)
    if wp.GetDistanceTo(ps.start) > ps.start.GetDistanceTo(ps.end):
        distance_to_circle = ps.max_radius + w.radius - wp.GetDistanceTo(ps.end)
    # debug_string += 'd_to_circle: %.0f\n' % distance_to_circle
    # debug_string += 'd_to_border1: %.0f\n' % wp.GetDistanceToSegment(ps.border1)
    # debug_string += 'd_to_border2: %.0f\n' % wp.GetDistanceToSegment(ps.border2)

    min_d = min(wp.GetDistanceToSegment(ps.border1), wp.GetDistanceToSegment(ps.border2),
                distance_to_circle)
    # debug_string += 'min_d:\n%.1f\ntime * max_speed:\n%.1f*%.1f=%.1f' % (
    #                    min_d,
    #                    time,
    #                    state.index[w.id].max_speed,
    #                    time * state.index[w.id].max_speed)
    # state.dbg_text(w, debug_string, RED)
    return min_d < time * state.index[w.id].max_speed
    

def CanHitWizard(me, w, action, state, strict=False):
    if action == ActionType.STAFF:
        return True
    ws = state.index[w.id]
    mes = state.index[me.id]
    t = ProjectileType.MAGIC_MISSILE
    r = state.game.magic_missile_radius
    if action == ActionType.FIREBALL:
        t = ProjectileType.FIREBALL
        r = state.game.fireball_explosion_min_damage_range
    if action == ActionType.FROST_BOLT:
        t = ProjectileType.FROST_BOLT
        r = state.game.frost_bolt_radius
    speed = Point.FromUnit(w) - Point.FromUnit(me)
    if strict:
        speed = Point(1.0, 0.0).Rotate(me.angle)
    fake_p = Projectile(id=-me.id, x=me.x, y=me.y, speed_x=speed.x, speed_y=speed.y,
        angle=speed.GetAngle(), faction=me.faction, radius=r, type=t, owner_unit_id=me.id,
        owner_player_id=me.owner_player_id)
    from State import ProjectileState
    ps = ProjectileState(p=fake_p, me=me, game=state.game,
                         world=state.world, last_state=state, dbg=state.dbg)
    return not CanDodge(w, ps, state)

def PickReachableTarget(me, cast_range, action, state, radius=INFINITY, lane=None):
    enemies = [e for e in BuildEnemies(me, radius, state) if 
               e.get_distance_to_unit(me) < cast_range + e.radius - CAST_RANGE_ERROR]
    if lane is not None:
        enemies = [e for e in enemies if (lane in GetLanes(e))]
    min_hp = INFINITY
    best_type = 0
    best = None
    for e in enemies:
        t = 0
        if isinstance(e, Wizard):
            if (radius < INFINITY) and (not CanHitWizard(me, e, action, state)):
                continue
            t = 2
        if isinstance(e, Building):
            t = 1
        if (t >= best_type) and ((t > best_type) or (e.life < min_hp) or 
                                 ((e.life == min_hp) and (e.id < best.id))):
            min_hp = e.life
            best = e
            best_type = t
    return best
    
def IsEnemy(u1, u2):
    return ((u1.faction != u2.faction) and (u1.faction != Faction.OTHER) and
            (u2.faction != Faction.OTHER) and (isinstance(u1, LivingUnit)) and
            (isinstance(u2, LivingUnit)) and (
             (u2.faction != Faction.NEUTRAL) or 
             (not NeutralMinionInactive(u2))))

def IsValidTarget(u1, u2, state):
    u2s = state.index[u2.id]
    if isinstance(u2, Building) and (u2.type == BuildingType.GUARDIAN_TOWER):
        towers_this_lane = 0
        for b in state.world.buildings:
            if u2s.is_on_top_lane and state.index[b.id].is_on_top_lane:
                towers_this_lane += 1
            if u2s.is_on_middle_lane and state.index[b.id].is_on_middle_lane:
                towers_this_lane += 1
            if u2s.is_on_bottom_lane and state.index[b.id].is_on_bottom_lane:
                towers_this_lane += 1
        if towers_this_lane > 1:
            return False
    return IsEnemy(u1, u2)


def PickTarget(me, action, state, radius=INFINITY, lane=None):
    best = PickReachableTarget(me, radius, action, state, radius, lane)
    return best
    
def PickMeleeTarget(me, state):
    return PickReachableTarget(me, state.game.staff_range, ActionType.STAFF, state)
    
def NeutralMinionInactive(m):
    return ((abs(m.speed_x) + abs(m.speed_y) < EPSILON) and
            m.life == m.max_life)

def GetAggro(me, safe_distance, state):
    allies = [a for a in state.world.wizards + state.world.minions + state.world.buildings if
              (a.id != me.id) and (a.faction == me.faction) and 
              (a.get_distance_to_unit(me) < me.vision_range * 1.5)]
    aggro = 0
    mes = state.index[me.id]
    
    #TODO(vyakunin): add aggro from respawn
    for e in state.world.wizards + state.world.minions + state.world.buildings:
        if (e.faction != me.faction):
            if not (e.id in state.index):
                import pdb; pdb.set_trace()
            es = state.index[e.id]
            d = es.dist
            deepness = (es.aggro_range + safe_distance + es.max_speed - d)
            if deepness > 0:
                aggro += GetUnitAggro(mes, es, deepness, state)
    aggro = mes.get_effective_damage_to_me(aggro) + mes.expected_overtime_damage 
    return aggro
    
def HaveEnoughTimeToTurn(w, angle, target, state, action=ActionType.MAGIC_MISSILE):
    ws = state.index[w.id]
    speed = state.game.wizard_max_turn_angle
    if StatusType.HASTENED in [st.type for st in w.statuses]:
        speed *= (1.0 + game.hastened_rotation_bonus_factor)
    return (max((abs(angle) / speed) + 2, 
                (-RangeAllowance(w, target, state)) / ws.strafe_speed) < 
            ws.remaining_action_cooldown(action))
            
class HistoricStateTracker(object):
    instance = None
    def __init__(self, me, world):
        self.buildings = []
        for b in world.buildings:
            nb = deepcopy(b)
            nb.faction = 1 - me.faction
            nb.remaining_action_cooldown_ticks = 0
            nb.id = -b.id - 1
            nb.x = world.width - b.x
            nb.y = world.height - b.y
            self.buildings.append(nb)
        self.last_fired = [0] * len(self.buildings)
        

    @classmethod
    def GetInstance(cls, me, world):
        if HistoricStateTracker.instance is None:
            HistoricStateTracker.instance = HistoricStateTracker(me, world)
        return HistoricStateTracker.instance
        
    def AddInvisibleBuildings(self, me, world, game):
        new_b = []
        dead_b = []
        bad_ids = []
        for i, b in enumerate(world.buildings):
            if b.id < 0:
                bad_ids.append(i)
        for i in reversed(bad_ids):
            world.buildings = world.buildings[:i] + world.buildings[i+1:]
        for i, fake_b in enumerate(self.buildings):
            found = False
            for real_b in world.buildings:
                if (real_b.faction != me.faction):
                    # if real_b.type == BuildingType.FACTION_BASE:
                    #     import pdb; pdb.set_trace()
                    if ((real_b.get_distance_to_unit(fake_b) < 500) and
                        (real_b.type == fake_b.type)):
                            found = True
                            fake_b.life = real_b.life
                            fake_b.remaining_action_cooldown_ticks = real_b.remaining_action_cooldown_ticks
                            fake_b.x = real_b.x
                            fake_b.y = real_b.y
                            self.last_fired[i] = world.tick_index - real_b.cooldown_ticks + real_b.remaining_action_cooldown_ticks
                            break
            if found:
                continue
            fake_b.remaining_action_cooldown_ticks = max(
                0, self.last_fired[i] + fake_b.cooldown_ticks - world.tick_index)
            for a in world.wizards + world.minions + world.buildings:
                if a.faction == me.faction:
                    if a.get_distance_to_unit(fake_b) < a.vision_range:
                        found = True
                        dead_b.append(i)
                        break
            if not found:
                new_b.append(fake_b)
        world.buildings.extend(new_b)
        for i in reversed(dead_b):
            self.buildings = self.buildings[:i] + self.buildings[i+1:]
            self.last_fired = self.last_fired[:i] + self.last_fired[i+1:]
            
def FindFirstUnitByIds(ids, state):
    for i in ids:
        if i in state.index:
            return state.index[i].unit