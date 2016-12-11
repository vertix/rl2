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
from Geometry import Segment
from Geometry import PlainCircle
from Geometry import IntersectCircles
from Geometry import IntersectSegments
from Geometry import PickDodgeDirectionAndTime
from math import sqrt

from Colors import RED
from Colors import GREEN
from Colors import BLUE

CAST_RANGE_ERROR = 0
EPSILON = 1e-4
INFINITY = 1e6
AGGRO_TICKS = 20
MAX_RANGE = 1000
MAX_TARGETS = 3
MACRO_EPSILON = 1

class TargetAndDamage(object):
    def __init__(self, target):
        self.target = target
        self.wizard_damage = 0.0
        self.building_damage = 0.0
        self.damage = 0.0

    def AddWizardDamage(self, d):
        self.wizard_damage += d

    def AddBuildingDamage(self, d):
        self.building_damage += d

    def AddDamage(self, d):
        self.damage += d

    def CombinedDamage(self, state):
        return (self.wizard_damage * (1 + state.game.wizard_damage_score_factor) +
                   self.building_damage * (1 + state.game.building_damage_score_factor) +
                   self.damage * (1 + state.game.minion_damage_score_factor))

def GetFireballDamage(mes, t, p, state):
    ws = None
    if isinstance(t, Wizard):
        ws = state.index[t.id]
        d = p.GetDistanceTo(mes.unit)
        time = d / state.game.fireball_speed
        final_distance = p.GetDistanceTo(t) - t.radius + MACRO_EPSILON
        if t.faction != mes.unit.faction:
             final_distance += time * ws.forward_speed
    else:
        final_distance = p.GetDistanceTo(t) - t.radius + MACRO_EPSILON
    if final_distance > state.game.fireball_explosion_min_damage_range:
        return 0
    if final_distance < state.game.fireball_explosion_max_damage_range:
        direct_damage = mes.fireball_direct_damage
    else:
        direct_damage = state.game.fireball_explosion_min_damage + (
            state.game.fireball_explosion_max_damage -
            state.game.fireball_explosion_min_damage) * (
            final_distance - state.game.fireball_explosion_max_damage_range) / (
            state.game.fireball_explosion_min_damage_range -
            state.game.fireball_explosion_max_damage_range)
        direct_damage = mes.get_effective_damage_by_me(direct_damage)
            
    if isinstance(t, Wizard):
        direct_damage = ws.get_effective_damage_to_me(direct_damage)
        direct_damage += (state.game.wizard_elimination_score_factor * 
                          (t.max_life - t.life + direct_damage + state.game.burning_summary_damage)
                          + state.game.burning_summary_damage)
    elif isinstance(t, Building):
        direct_damage += (state.game.building_elimination_score_factor * 
                          (t.max_life - t.life + direct_damage + state.game.burning_summary_damage)
                          + state.game.burning_summary_damage)
    else:
        direct_damage = min(t.life, direct_damage + state.game.burning_summary_damage)
    return direct_damage

def PickBestFireballTarget(me, state):
    mes = state.index[me.id]
    if mes.fireball == 0 or mes.fireball_cooldown > 2:
        return None
    sorted_targets = [deepcopy(t) for t in sorted(BuildTargets(
                       me, me.cast_range +
                       state.game.fireball_explosion_min_damage_range +
                       state.game.faction_base_radius, state), key=lambda x: (
                        -3*isinstance(x, Wizard) - 2*isinstance(x, Building),
                        -state.index[x.id].damage, -state.index[x.id].hp))]
    targets = []
    for t in sorted_targets:
        t.radius += state.game.fireball_explosion_min_damage_range - MACRO_EPSILON
        targets.append(t)
    candidates = []
    segments = []
    for i, t in enumerate(targets[:MAX_TARGETS]):
        state.dbg_circle(t, RED)
        candidates.append(Point.FromUnit(t))
        for j in range(i+1, len(targets[:MAX_TARGETS])):
            t2 = targets[j]
            intersections = IntersectCircles(t, t2)
            if intersections is not None:
                p1 = intersections[0][0]
                p2 = intersections[0][1]
                candidates.append((p1 + p2) * 0.5)
                if p1.GetSqDistanceTo(p2) > EPSILON:
                    segments.append(Segment(p1, p2))
    for i, s1 in enumerate(segments):
        state.dbg_line(s1.p1, s1.p2, RED)
        for j in range(i + 1, len(segments)):
            s2 = segments[j]
            intersection = IntersectSegments(s1, s2)
            if intersection is not None:
                candidates.append(intersection)
    best = None
    friend_wizards = [w for w in state.world.wizards if w.faction == me.faction]
    for t in targets:
        t.radius -= state.game.fireball_explosion_min_damage_range - MACRO_EPSILON

    for c in candidates:
        d = c.GetDistanceTo(me)
        if d > me.cast_range:
            c = Point.FromUnit(me) + (c - me) * (me.cast_range / d)
            d = c.GetDistanceTo(me)
        state.dbg_circle(PlainCircle(c, 3), GREEN)
        if d < state.game.fireball_explosion_min_damage_range + me.radius + mes.max_speed + MACRO_EPSILON:
            c = Point.FromUnit(me) + (c - me) * ((
                state.game.fireball_explosion_min_damage_range + me.radius + mes.max_speed + MACRO_EPSILON) / d)
            d = c.GetDistanceTo(me)
        res = TargetAndDamage(c)
        for t in targets:
            damage = GetFireballDamage(mes, t, c, state)
            if damage > 0:
                if isinstance(t, Wizard):
                    res.AddWizardDamage(damage)
                elif isinstance(t, Building):
                    res.AddBuildingDamage(damage)
                else:
                    res.AddDamage(damage)
        for f in friend_wizards:
            damage = GetFireballDamage(mes, f, c, state)
            if damage > 0:
                if isinstance(t, Wizard):
                    res.AddWizardDamage(-damage)
                elif isinstance(t, Building):
                    res.AddBuildingDamage(-damage)
                else:
                    res.AddDamage(-damage)
        combined_damage = res.CombinedDamage(state)
        state.dbg_text(c, '\n\n\n%d' % combined_damage)
        if combined_damage > 0 and (best is None or (
            combined_damage > 
            best.CombinedDamage(state))):
            best = res
    if best is not None:
        state.dbg_circle(PlainCircle(best.target, 5), BLUE)
        if best.CombinedDamage(state) < 10:
            return None
    return best

def GetAggroFromDamage(damage, remaining_cooldown, cooldown, deepness):
    return (max(0, int(AGGRO_TICKS + deepness - remaining_cooldown + cooldown - 1)) /
            int(cooldown) * damage)

def GetUnitAggro(mes, us, deep_in_range, state):
    aggro = 0.0
    speed = mes.forward_speed
    runaway_speed = max(1.0, speed - us.max_speed)
    time_to_leave = deep_in_range / runaway_speed
    if isinstance(us.unit, Wizard):
        # TODO(vyakunin): this should increase TTL for all units!
        if (us.frost_bolt > 0 and us.frost_bolt_cooldown < time_to_leave):
            time_to_leave += state.game.frozen_duration_ticks
        
        aggro = GetAggroFromDamage(us.fireball,
                         us.fireball_cooldown,
                         us.fireball_total_cooldown,
                         time_to_leave)
        aggro += GetAggroFromDamage(us.frost_bolt,
                         us.frost_bolt_cooldown,
                         us.frost_bolt_total_cooldown,
                         time_to_leave)
        aggro += GetAggroFromDamage(us.missile,
                         us.missile_cooldown,
                         us.missile_total_cooldown,
                         time_to_leave)
        if us.dist < state.game.staff_range:
            aggro += GetAggroFromDamage(us.staff,
                                        us.staff_cooldown,
                                        state.game.staff_cooldown_ticks,
                                        time_to_leave)
    else:
        aggro = GetAggroFromDamage(us.damage, us.cooldown_ticks, 
                                   us.total_cooldown_ticks,
                                   time_to_leave)
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
        # if u.id == -21:
        #     import pdb; pdb.set_trace()
        d = xp.GetSqDistanceTo(u)
        if res > d:
            res = d
            a = u 
    return a
    
def GetClosestTarget(me, state, radius=MAX_RANGE):
    return ClosestUnit(me, BuildTargets(me, min(radius, me.vision_range*2), state))
    
def BuildEnemies(me, radius, state):
    return [e for e in state.world.wizards + state.world.minions + state.world.buildings if
                   IsEnemy(me, e) 
                   and (e.get_distance_to_unit(me) < radius)]

def BuildTargets(me, radius, state):
    return [t for t in BuildEnemies(me, radius, state) if IsValidTarget(me, t, state)]

def CanDodge(w, ps, state, projectile_radius_modifier=-MACRO_EPSILON, fine_tune=True):
    debug_string = ''
    wp = Point.FromUnit(w)
    if (wp.GetDistanceToSegment(ps.center_line) >
        ps.max_radius + projectile_radius_modifier + w.radius):
        return True
    psp = Point.FromUnit(ps.p)
    projectile_distance = ps.start.GetDistanceTo(ps.end)
    d = min(projectile_distance, wp.ProjectToLine(ps.center_line.l).GetDistanceTo(ps.start))
    state.dbg_circle(PlainCircle(ps.end, ps.max_radius + w.radius))
    state.dbg_line(ps.border1.p1, ps.border1.p2)
    state.dbg_line(ps.border2.p1, ps.border2.p2)
    time = d / ps.speed
    distance_to_circle = INFINITY
    debug_string += 'w_to_start: %.0f\n' % wp.GetDistanceTo(ps.start)
    debug_string += 'start_to_end: %.0f\n' % ps.start.GetDistanceTo(ps.end)
    if ((ps.end - ps.start).ScalarMul(wp - ps.start) > 0
        and (ps.end - ps.start).ScalarMul(wp - ps.end) > 0):
        distance_to_circle = (ps.max_radius + projectile_radius_modifier +
                              w.radius - wp.GetDistanceTo(ps.end))
        
    debug_string += 'd_to_circle: %.0f\n' % distance_to_circle
    debug_string += 'd_to_border1: %.0f\n' % wp.GetDistanceToSegment(ps.border1)
    debug_string += 'd_to_border2: %.0f\n' % wp.GetDistanceToSegment(ps.border2)

    min_d = min(wp.GetDistanceToSegment(ps.border1) + projectile_radius_modifier,
                wp.GetDistanceToSegment(ps.border2) + projectile_radius_modifier,
                distance_to_circle)
    debug_string += 'min_d:\n%.1f\ntime * max_speed:\n%.1f*%.1f=%.1f' % (
                       min_d,
                       time,
                       state.index[w.id].max_speed,
                       time * state.index[w.id].max_speed)
    state.dbg_text(w, debug_string, RED)
    if min_d > time * state.index[w.id].max_speed:
        return False
    if fine_tune:
        return PickDodgeDirectionAndTime(w, ps, state, projectile_radius_modifier) is not None
    return True
    
def ActionToProjectileTypeAndRadius(a, state):
    if a == ActionType.FIREBALL:
        return (ProjectileType.FIREBALL, state.game.fireball_explosion_min_damage_range)
    if a == ActionType.FROST_BOLT:
        return (ProjectileType.FROST_BOLT, state.game.frost_bolt_radius)
    return (ProjectileType.MAGIC_MISSILE, state.game.magic_missile_radius)

def CanHitWizard(me, w, action, state, strict=False):
    if action == ActionType.STAFF:
        return True
    ws = state.index[w.id]
    mes = state.index[me.id]
    t, r = ActionToProjectileTypeAndRadius(action, state)
    
    speed = Point.FromUnit(w) - Point.FromUnit(me)
    if strict:
        da = me.get_angle_to_unit(w)
        if abs(da) > mes.max_rotation_speed:
            da = da / abs(da) * mes.max_rotation_speed
            speed = Point(1.0, 0.0).Rotate(me.angle + da)
    fake_p = Projectile(id=-me.id-1, x=me.x, y=me.y, speed_x=speed.x, speed_y=speed.y,
        angle=speed.GetAngle(), faction=me.faction, radius=r, type=t, owner_unit_id=me.id,
        owner_player_id=me.owner_player_id)
    from State import ProjectileState
    ps = ProjectileState(p=fake_p, me=me, game=state.game,
                         world=state.world, state=state, dbg=state.dbg)
    return not CanDodge(w, ps, state)

def PickReachableTarget(me, cast_range, action, state, radius=MAX_RANGE, lane=None):
    enemies = [e for e in BuildTargets(me, radius, state) if 
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
    if (isinstance(u2, Building) and (u2.type == BuildingType.GUARDIAN_TOWER)
        and (u2.x > 2600 and u2.y < 1400)):
        # if not u2s.is_on_middle_lane:
        #     import pdb; pdb.set_trace()
        
        towers_this_lane = 0
        for b in state.world.buildings:
            if (b.faction != u2.faction) or (b.type != BuildingType.GUARDIAN_TOWER):
                continue
            if u2s.is_on_top_lane and state.index[b.id].is_on_top_lane:
                towers_this_lane += 1
            if u2s.is_on_middle_lane and state.index[b.id].is_on_middle_lane:
                towers_this_lane += 1
            if u2s.is_on_bottom_lane and state.index[b.id].is_on_bottom_lane:
                towers_this_lane += 1
        if towers_this_lane > 1:
            return False
    return IsEnemy(u1, u2)


def PickTarget(me, action, state, radius=MAX_RANGE, lane=None):
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
            # if not (e.id in state.index):
            #     import pdb; pdb.set_trace()
            es = state.index[e.id]
            d = es.dist
            deepness = (es.aggro_range + safe_distance + es.max_speed - d)
            if deepness > 0:
                aggro += GetUnitAggro(mes, es, deepness, state)
    aggro = mes.get_effective_damage_to_me(aggro) 
    return aggro
    
def HaveEnoughTimeToTurn(w, angle, target, state, action=ActionType.MAGIC_MISSILE):
    ws = state.index[w.id]
    speed = ws.max_rotation_speed
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
                    d = a.get_distance_to_unit(fake_b)
                    if d < a.vision_range:
                        found = True
                        dead_b.append(i)
                        break
                    if d <= fake_b.attack_range and fake_b.remaining_action_cooldown_ticks == 0:
                        fake_b.remaining_action_cooldown_ticks = fake_b.cooldown_ticks - 1
                        self.last_fired[i] = world.tick_index
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