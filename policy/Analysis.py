from model.Faction import Faction
from model.MinionType import MinionType
from model.Building import Building


WIZARD = 100
WOODCUTTER = 20
FETISH = 40
TOWER = 150

def Closest(x, units):
    res = 1e6
    for u in units:
        d = u.get_distance_to_unit(x)
        res = min(res, d)
    return res
    
def PickTarget(me, world, game):
    enemies = [e for e in world.wizards + world.minions + world.buildings if
               (e.faction != me.faction) and (e.faction != Faction.NEUTRAL) and 
               (e.faction != Faction.OTHER) and 
               (e.get_distance_to_unit(me) < me.cast_range - e.radius)]
    min_hp = 1e6
    best = None
    for e in enemies:
        if isinstance(e, Building):
            return e
        if e.life < min_hp:
            min_hp = e.life
            best = e
    return best

def GetAggro(me, game, world):
    allies = [a for a in world.wizards + world.minions + world.buildings if
              (a.id != me.id) and (a.faction == me.faction) and 
              (a.get_distance_to_unit(me) < me.vision_range)]
    aggro = 0
    for w in world.wizards:
        d = w.get_distance_to_unit(me)
        if w.faction != me.faction and d - me.radius < w.cast_range:
            aggro += WIZARD
    for m in world.minions:
        if (m.faction != me.faction and m.faction != Faction.NEUTRAL and 
            m.faction != Faction.OTHER):
            d = m.get_distance_to_unit(me)
            if m.type == MinionType.ORC_WOODCUTTER:
                if d - me.radius < game.orc_woodcutter_attack_range:
                    if Closest(m, allies) >= d:
                        aggro += WOODCUTTER
            else:
                if d - me.radius < game.fetish_blowdart_attack_range:
                    if Closest(m, allies) >= d:
                        aggro += FETISH
    for b in world.buildings:
        d = b.get_distance_to_unit(me)
        if b.faction != me.faction and d - me.radius < b.attack_range:
            if Closest(b, allies) >= d:
                aggro += TOWER
    return aggro