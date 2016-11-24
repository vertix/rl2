from model.Faction import Faction
from model.MinionType import MinionType
from model.Building import Building
from model.Minion import Minion
from model.Wizard import Wizard
from model.ActionType import ActionType
from model.StatusType import StatusType
from model.BuildingType import BuildingType
from Geometry import RangeAllowance


WIZARD = 100
WOODCUTTER = 20
FETISH = 40
TOWER = 70
BASE = 100
CAST_RANGE_ERROR = 5
EPSILON = 1e-4
INFINITY = 1e6
AGGRO_TICKS = 50

def BuildCoeff(coeff):
    return max(0.5, coeff)

def GetAggroFromDamage(damage, remaining_cooldown, cooldown, deepness):
    return max(0, (AGGRO_TICKS + deepness - remaining_cooldown + cooldown - 1) /
                  cooldown * damage)

def GetUnitAggro(me, u, game, deep_in_range):
    aggro = 0
    speed = GetMaxStrafeSpeed(me, game)
    if isinstance(u, Wizard):
        aggro = GetAggroFromDamage(GetWizardDamage(u, game),
                         max(u.remaining_action_cooldown_ticks,
                             u.remaining_cooldown_ticks_by_action[ActionType.MAGIC_MISSILE]),
                         game.magic_missile_cooldown_ticks, 
                         deep_in_range / (speed - GetMaxStrafeSpeed(u, game) / 2.0))
        if me.get_distance_to_unit(u) < game.staff_range:
            aggro += GetAggroFromDamage(GetWizardDamage(u, game),
                         max(u.remaining_action_cooldown_ticks,
                             u.remaining_cooldown_ticks_by_action[ActionType.STAFF]),
                         game.staff_cooldown_ticks)
    if isinstance(u, Building) or isinstance(u, Minion):
        aggro = GetAggroFromDamage(u.damage, u.remaining_action_cooldown_ticks, 
                                   u.cooldown_ticks,
                                   deep_in_range / speed)
    return aggro

def Closest(x, units):
    res = INFINITY
    for u in units:
        d = u.get_distance_to_unit(x)
        res = min(res, d)
    return res
    
def ClosestUnit(x, units):
    res = INFINITY
    a = None
    for u in units:
        d = u.get_distance_to_unit(x)
        if res > d:
            res = d
            a = u 
    return a
    
def GetClosestTarget(me, world, game, radius=INFINITY):
    return ClosestUnit(me, BuildEnemies(me, world, game, min(radius, me.vision_range*2)))
    
def BuildEnemies(me, world, game, radius):
    return [e for e in world.wizards + world.minions + world.buildings if
                   IsEnemy(me, e) 
                   and (e.get_distance_to_unit(me) < radius)]

def PickReachableTarget(me, world, game, radius=INFINITY):
    enemies = [e for e in BuildEnemies(me, world, game, radius) if 
               e.get_distance_to_unit(me) < me.cast_range + e.radius - CAST_RANGE_ERROR]
    min_hp = INFINITY
    best_type = 0
    best = None
    for e in enemies:
        t = 0
        if isinstance(e, Wizard):
            t = 2
        if isinstance(e, Building):
            t = 1
        if (t >= best_type) and ((t > best_type) or (e.life < min_hp) or 
                                 ((e.life == min_hp) and (e.id < best.id))):
            min_hp = e.life
            best = e
            best_type = t
    return best
    
def IsEnemy(me, e):
    return (e.faction != me.faction) and (e.faction != Faction.OTHER) and (
            (e.faction != Faction.NEUTRAL) or (not NeutralMinionInactive(e)))

def PickTarget(me, world, game, radius=INFINITY):
    best = PickReachableTarget(me, world, game, radius)
    if best is None:
        return GetClosestTarget(me, world, game, radius)
    return best
    
def PickMeleeTarget(me, world, game):
    return PickTarget(me, world, game, game.staff_range)
    
def NeutralMinionInactive(m):
    return ((abs(m.speed_x) + abs(m.speed_y) < EPSILON) and
            m.life == m.max_life)

def GetMinionAggro(me, allies, m, game, world, safe_distance):
    if ((m.faction == me.faction) or 
        ((m.faction == Faction.NEUTRAL) and NeutralMinionInactive(m)) or 
        (m.faction == Faction.OTHER)):
        return 0

    d = m.get_distance_to_unit(me)
    if d > m.vision_range + EPSILON:
        return 0
    if m.type == MinionType.ORC_WOODCUTTER:
        deepness = min(Closest(m, allies),
                       game.orc_woodcutter_attack_range) + safe_distance - d
        if deepness > 0:
            return GetUnitAggro(me, m, game, deepness)
    deepness = min(Closest(m, allies),
                   game.fetish_blowdart_attack_range + me.radius) + safe_distance - d
    if deepness > 0:
        return GetUnitAggro(me, m, game, deepness)
    return 0

def GetAggro(me, game, world, safe_distance):
    allies = [a for a in world.wizards + world.minions + world.buildings if
              (a.id != me.id) and (a.faction == me.faction) and 
              (a.get_distance_to_unit(me) < me.vision_range * 2)]
    aggro = 0
    
    #TODO(vyakunin): add aggro from respawn
    for w in world.wizards:
        d = w.get_distance_to_unit(me)
        deepness = w.cast_range + safe_distance - d + me.radius
        if (w.faction != me.faction) and (deepness > 0):
            aggro += GetUnitAggro(me, w, game, deepness)
    for m in world.minions:
        aggro += GetMinionAggro(me, allies, m, game, world, safe_distance)
    for b in world.buildings:
        d = b.get_distance_to_unit(me)
        deepness = b.attack_range + safe_distance - d
        if (b.faction != me.faction) and (deepness > 0):
            aggro += GetUnitAggro(me, b, game, deepness)
    if StatusType.SHIELDED in [s.type for s in me.statuses]:
        aggro *= (1.0-game.shielded_direct_damage_absorption_factor) 
    return aggro
    
def GetRemainingActionCooldown(w, action=ActionType.MAGIC_MISSILE):
    return max(w.remaining_action_cooldown_ticks, 
               w.remaining_cooldown_ticks_by_action[action])
               
def HaveEnoughTimeToTurn(w, angle, target, game, action=ActionType.MAGIC_MISSILE):
    speed = game.wizard_max_turn_angle
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        speed *= game.hastened_rotation_bonus_factor
    return (max((abs(angle) / speed) + 2, 
                (-RangeAllowance(w, target)) / GetMaxStrafeSpeed(w, game)) < 
            GetRemainingActionCooldown(w, action))
            
def GetWizardDamage(w, game):
    d = game.staff_damage
    if StatusType.EMPOWERED in [s.type for s in w.statuses]:
        d *= game.empowered_damage_factor
    return d
    
def GetMaxForwardSpeed(w, game):
    s = game.wizard_forward_speed
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        s *= game.hastened_movement_bonus_factor
    return s
    
def GetMaxStrafeSpeed(w, game):
    s = game.wizard_strafe_speed
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        s *= game.hastened_movement_bonus_factor
    return s

def FindUnitById(world, t_id):
    for o in world.wizards + world.buildings + world.minions + world.trees:
        if t_id == o.id:
            return o
    return None