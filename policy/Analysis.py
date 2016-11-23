from model.Faction import Faction
from model.MinionType import MinionType
from model.Building import Building
from model.ActionType import ActionType
from model.StatusType import StatusType
from Geometry import RangeAllowance


WIZARD = 100
WOODCUTTER = 20
FETISH = 40
TOWER = 150
CAST_RANGE_ERROR = 5

def Closest(x, units):
    res = 1e6
    for u in units:
        d = u.get_distance_to_unit(x)
        res = min(res, d)
    return res
    
def ClosestUnit(x, units):
    res = 1e6
    a = None
    for u in units:
        d = u.get_distance_to_unit(x)
        if res > d:
            res = d
            a = u 
    return a
    
def PickReachableTarget(me, world, game):
    enemies = [e for e in world.wizards + world.minions + world.buildings if
               (e.faction != me.faction) and (e.faction != Faction.NEUTRAL) and 
               (e.faction != Faction.OTHER) and (
                e.get_distance_to_unit(me) < me.cast_range + e.radius - CAST_RANGE_ERROR)] 
    min_hp = 1e6
    best = None
    for e in enemies:
        if isinstance(e, Building):
            return e
        if (e.life < min_hp) or ((e.life == min_hp) and (e.id < best.id)):
            min_hp = e.life
            best = e
    return best

def PickTarget(me, world, game):
    best = PickReachableTarget(me, world, game)
    if best is None:
        enemies = [e for e in world.wizards + world.minions + world.buildings if
                   (e.faction != me.faction) and (e.faction != Faction.NEUTRAL) and 
                   (e.faction != Faction.OTHER) and (e.get_distance_to_unit(me) < 700)]

        return ClosestUnit(me, enemies)
    return best


def GetAggro(me, game, world, safe_distance):
    allies = [a for a in world.wizards + world.minions + world.buildings if
              (a.id != me.id) and (a.faction == me.faction) and 
              (a.get_distance_to_unit(me) < me.vision_range)]
    aggro = 0
    for w in world.wizards:
        d = w.get_distance_to_unit(me)
        if (w.faction != me.faction) and (d - me.radius < w.cast_range + safe_distance):
            aggro += WIZARD
    for m in world.minions:
        if ((m.faction != me.faction) and (m.faction != Faction.NEUTRAL) and 
            (m.faction != Faction.OTHER)):
            d = m.get_distance_to_unit(me)
            if m.type == MinionType.ORC_WOODCUTTER:
                if Closest(m, allies) > d - safe_distance:
                    aggro += WOODCUTTER
            else:
                if d - me.radius < game.fetish_blowdart_attack_range + safe_distance:
                    if Closest(m, allies) > d - safe_distance:
                        aggro += FETISH
    for b in world.buildings:
        d = b.get_distance_to_unit(me)
        if (b.faction != me.faction) and (d - me.radius < b.attack_range + safe_distance):
            if Closest(b, allies) > d - safe_distance:
                aggro += TOWER
            else:
                aggro += TOWER / 2
    return aggro
    
def GetRemainingActionCooldown(w, action=ActionType.MAGIC_MISSILE):
    return max(w.remaining_action_cooldown_ticks, 
               w.remaining_cooldown_ticks_by_action[action])
               
def HaveEnoughTimeToTurn(w, angle, target, game, action=ActionType.MAGIC_MISSILE):
    speed = game.wizard_max_turn_angle
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        speed *= hastened_rotation_bonus_factor
    return (max((abs(angle) / speed) + 2, 
               RangeAllowance(w, target) / GetMaxStrafeSpeed(w, game)) 
            < GetRemainingActionCooldown(w, action))
    
def GetMaxForwardSpeed(w, game):
    s = game.wizard_forward_speed
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        s *= hastened_movement_bonus_factor
    return s
    
def GetMaxStrafeSpeed(w, game):
    s = game.wizard_strafe_speed
    if StatusType.HASTENED in [s.type for s in w.statuses]:
        s *= hastened_movement_bonus_factor
    return s