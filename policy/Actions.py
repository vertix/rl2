from model.ActionType import ActionType
from model.Game import Game
from model.LaneType import LaneType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.Unit import Unit
from model.Message import Message
from model.SkillType import SkillType
from model.Building import Building
from model.Projectile import Projectile
from model.ProjectileType import ProjectileType
from Colors import RED
from Geometry import Point
from Geometry import HasMeleeTarget
from Geometry import TargetInRangeWithAllowance
from Geometry import ProjectileWillHit
from Geometry import PickDodgeDirectionAndTime
from Geometry import PlainCircle
from CachedGeometry import Cache
from Analysis import GetAggro
from Analysis import CanHitWizard
from Analysis import CanDodge
from Analysis import PickTarget
from Analysis import PickReachableTarget
from Analysis import HaveEnoughTimeToTurn
from Analysis import FindFirstUnitByIds
from Analysis import PickMeleeTarget
from Analysis import GetClosestTarget
from copy import deepcopy

import math


NUM_STEPS_PER_LANE = 7
WAYPOINT_RADIUS = 300 # must be less than distance between waypoints!
TARGET_COOLDOWN = 20
MISSILE_DISTANCE_ERROR = 10
MARGIN = 300
INFINITY=1e6
EPSILON = 1e-5
MACRO_EPSILON = 1
last_x = -1
last_y = -1

class NoOpAction(object):
    def Act(self, me, state):
        return Move()

    @property
    def name(self):
        return "NOOP"


def GetNextWaypoint(waypoints, me):
    # import pdb; pdb.set_trace()
    last_point = waypoints[-1]
    for i, point in enumerate(waypoints):
        if me.get_distance_to_unit(point) < WAYPOINT_RADIUS:
            return min(i + 1, len(waypoints) - 1)
        if point.GetDistanceTo(last_point) < me.get_distance_to_unit(last_point):
            return i
    return len(waypoints) - 1


def GetPrevWaypoint(waypoints, me):
    first_point = waypoints[0]
    for i, point in reversed(list(enumerate(waypoints))):
        if i == 0:
            return 0
        if me.get_distance_to_unit(point) < WAYPOINT_RADIUS:
            return i-1
        if point.GetDistanceTo(first_point) < me.get_distance_to_unit(first_point):
            return i
    return 0


class MoveAction(object):
    def __init__(self, map_size, lane):
        self.lane = lane
        step = map_size / NUM_STEPS_PER_LANE
        start = Point(MARGIN, map_size - MARGIN)
        end = Point(map_size - 100, 100)
        self.waypoints_by_lane = {
            LaneType.MIDDLE: [Point(i * step, map_size - i * step)
                              for i in range(3, NUM_STEPS_PER_LANE - 4)] + 
                             [Point(i * step + 200, map_size - i * step)
                              for i in range(NUM_STEPS_PER_LANE - 4, NUM_STEPS_PER_LANE - 2)],
            LaneType.TOP: [Point(MARGIN, map_size - i * step)
                           for i in range(3, NUM_STEPS_PER_LANE)] +
                          [Point(i * step, MARGIN) for i in range(1, NUM_STEPS_PER_LANE - 2)],
            LaneType.BOTTOM: [Point(i * step, map_size - MARGIN)
                              for i in range(3, NUM_STEPS_PER_LANE)] +
                             [Point(map_size - MARGIN, map_size - i * step)
                              for i in range(1, NUM_STEPS_PER_LANE - 4)] +
                             [Point(map_size - 100, map_size - i * step) 
                              for i in range(NUM_STEPS_PER_LANE - 4, NUM_STEPS_PER_LANE - 2)] 
        }

        for key in self.waypoints_by_lane:
            self.waypoints_by_lane[key] = [start] + self.waypoints_by_lane[key] + [end]
        self.focus_target = None
        self.last_target = 0
        self.overridden_target = None
        self.dodging = False
        
    def MoveTowardsAngle(self, me, angle, move, d, state):
        mes = state.index[me.id]
        if abs(angle) < math.pi / 2:
            move.speed = math.cos(angle) * mes.max_speed
        else:
            move.speed = math.cos(angle) * mes.max_speed
        move.strafe_speed = math.sin(angle) * mes.max_speed
        vd = math.hypot(move.speed, move.strafe_speed)
        divisor = 1.0
        if (abs(vd) > d) and (abs(d) > 1e-3):
            divisor = abs(vd / d)
        divisor = max(divisor, math.hypot(move.speed / mes.forward_speed,
                                          move.strafe_speed / mes.strafe_speed))
        move.speed /= divisor
        move.strafe_speed /= divisor
        # if abs(move.speed) > mes.forward_speed:
        #     import pdb; pdb.set_trace()
        # state.dbg_text(me, 'd:%.0f\ns:%.1f\nss:%.1f\nx:%.0f\ny:%.0f\nangle:%.1frel_angle:%.1f' % (
        #     d, move.speed, move.strafe_speed, me.x, me.y, me.angle, angle))
        new_p = Point(move.speed, move.strafe_speed).Rotate(me.angle) + me
        if not self.dodging:
            for p in state.world.projectiles:
                if ProjectileWillHit(state.index[p.id], me):
                    continue
                shifted_me = PlainCircle(new_p, me.radius)
                shifted_me.faction = me.faction
                if ProjectileWillHit(state.index[p.id], shifted_me):
                    move.speed = 0.0
                    move.strafe_speed = 0.0
                    new_p = me
                    break 
        state.dbg_line(me, new_p)
        state.dbg_line(me, Point(100.0, 0.0).Rotate(me.angle) + me, RED)
    
    def MaybeSetLanes(self, me, move):
        if me.master:
            move.messages = [Message(LaneType.MIDDLE, None, ''),
                             Message(LaneType.TOP, None, ''),
                             Message(LaneType.TOP, None, ''),
                             Message(LaneType.BOTTOM, None, '')]

    def LearnRange(self, me, move):
        if not SkillType.RANGE_BONUS_PASSIVE_1 in me.skills:
            move.skill_to_learn = SkillType.RANGE_BONUS_PASSIVE_1
            return True
        if not SkillType.RANGE_BONUS_AURA_1 in me.skills:
            move.skill_to_learn = SkillType.RANGE_BONUS_AURA_1
            return True
        if not SkillType.RANGE_BONUS_PASSIVE_2 in me.skills:
            move.skill_to_learn = SkillType.RANGE_BONUS_PASSIVE_2
            return True
        if not SkillType.RANGE_BONUS_AURA_2 in me.skills:
            move.skill_to_learn = SkillType.RANGE_BONUS_AURA_2
            return True
        if not SkillType.ADVANCED_MAGIC_MISSILE in me.skills:
            move.skill_to_learn = SkillType.ADVANCED_MAGIC_MISSILE
            return True
        return False

    def LearnDamage(self, me, move):
        if not SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_1 in me.skills:
            move.skill_to_learn = SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_1
            return True
        if not SkillType.MAGICAL_DAMAGE_BONUS_AURA_1 in me.skills:
            move.skill_to_learn = SkillType.MAGICAL_DAMAGE_BONUS_AURA_1
            return True
        if not SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_2 in me.skills:
            move.skill_to_learn = SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_2
            return True
        if not SkillType.MAGICAL_DAMAGE_BONUS_AURA_2 in me.skills:
            move.skill_to_learn = SkillType.MAGICAL_DAMAGE_BONUS_AURA_2
            return True
        if not SkillType.FROST_BOLT in me.skills:
            move.skill_to_learn = SkillType.FROST_BOLT
            return True
        return False

    def LearnFireball(self, me, move):
        if not SkillType.STAFF_DAMAGE_BONUS_PASSIVE_1 in me.skills:
            move.skill_to_learn = SkillType.STAFF_DAMAGE_BONUS_PASSIVE_1
            return True
        if not SkillType.STAFF_DAMAGE_BONUS_AURA_1 in me.skills:
            move.skill_to_learn = SkillType.STAFF_DAMAGE_BONUS_AURA_1
            return True
        if not SkillType.STAFF_DAMAGE_BONUS_PASSIVE_2 in me.skills:
            move.skill_to_learn = SkillType.STAFF_DAMAGE_BONUS_PASSIVE_2
            return True
        if not SkillType.STAFF_DAMAGE_BONUS_AURA_2 in me.skills:
            move.skill_to_learn = SkillType.STAFF_DAMAGE_BONUS_AURA_2
            return True
        if not SkillType.FIREBALL in me.skills:
            move.skill_to_learn = SkillType.FIREBALL
            return True
        return False

    def MaybeLearnSkills(self, me, move):
        if self.LearnFireball(me, move):
            return
        if self.LearnRange(me, move):
            return
        if self.LearnDamage(me, move):
            return

    def MaybeDodge(self, move, state):
        for p in state.world.projectiles:
            ps = state.index[p.id]
            me = state.my_state.unit
            if (ProjectileWillHit(ps, me) and
                CanDodge(me, ps, state, MACRO_EPSILON, fine_tune=False)):
                t = PickDodgeDirectionAndTime(me, ps, state, MACRO_EPSILON)
                if t is not None:
                    self.dodging = True
                    state.dbg_line(me, t.direction, RED)
                    self.RushToTarget(me, t.direction, move, state)
                    return True
        return False
        

    def RushToTarget(self, me, target, move, state):
        # if target.x > 3800 and target.y < 1400 and hasattr(target, 'id') and target.id < 0:
        #     import pdb; pdb.set_trace()
        mes = state.index[me.id]
        angle = None
        d = min(me.get_distance_to_unit(target), mes.max_speed)
        path = None
        t_ids = []
        if not self.dodging:
            path = Cache.GetInstance().GetPathToTarget(me, target, state)
            path.Show(state)
        if path is None:
            angle = me.get_angle_to_unit(target)
        else:
            out_tuple = path.GetNextAngleDistanceAndTargets(me)
            if out_tuple is not None:
                angle, new_d, t_ids = out_tuple
                if new_d < d:
                    d = new_d
        if angle is None:
            angle = me.get_angle_to_unit(target)
        self.MoveTowardsAngle(me, angle, move, d, state)
        # if t_ids:
        #     import pdb; pdb.set_trace()
        self.overridden_target = FindFirstUnitByIds(t_ids, state)

        move.turn = angle
        return None

    def MakeFleeMove(self, me, move, state):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetPrevWaypoint(waypoints, me)
        target = waypoints[i]
        # print Point.FromUnit(me), target
        self.RushToTarget(me, target, move, state)
            

    def MakeAdvanceMove(self, me, move, state):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetNextWaypoint(waypoints, me)
        target = waypoints[i]
        self.RushToTarget(me, target, move, state)
    
    def PickActionAndCooldown(self, me, target, state):
        mes = state.index[me.id]
        if target is None:
            return (ActionType.MAGIC_MISSILE, 
                    mes.missile_cooldown,
                    state.game.magic_missile_radius)
        ts = state.index[target.id]
        if (isinstance(target, Wizard) or
            (me.mana > me.max_mana * 0.8) or
            isinstance(target, Building)) and (
                target.life - ts.expected_overtime_damage > 
                ts.get_effective_damage_to_me(mes.missile)):
            if (mes.frost_bolt > 0 and mes.frost_bolt_cooldown == 0):
                return (ActionType.FROST_BOLT, 
                        mes.frost_bolt_cooldown,
                        state.game.frost_bolt_radius)
        return (ActionType.MAGIC_MISSILE, 
                mes.missile_cooldown,
                state.game.magic_missile_radius)

    def MakeMissileMove(self, me, move, state, target=None):
        mes = state.index[me.id]
        action, cooldown, radius = self.PickActionAndCooldown(me, target, state)
        t = None
        if target is not None:
            t = deepcopy(target)
        if t is None:
            t = PickReachableTarget(me, me.cast_range, action, state)
        
        if self.overridden_target is not None:
            if ((t is None) or
                (me.get_distance_to_unit(self.overridden_target) < state.game.staff_range + 1)):
                t = deepcopy(self.overridden_target)
        if t is None:
            return
        distance = me.get_distance_to_unit(t)

        if cooldown == 0:
            move.action = action
        if (not TargetInRangeWithAllowance(me, t, -radius, state) or
            (isinstance(t, Wizard) and 
             not CanHitWizard(me, t, action, state, True))):
            n_t = PickReachableTarget(me, me.cast_range, action, state)
            if n_t is not None:
                t = n_t
                distance = me.get_distance_to_unit(t)
                if (not TargetInRangeWithAllowance(me, t, -radius, state) or
                    (isinstance(t, Wizard) and 
                     not CanHitWizard(me, t, action, state, True))):
                    move.action = ActionType.NONE
            else:
                move.action = ActionType.NONE
        move.min_cast_distance = min(
            me.cast_range, 
            me.get_distance_to_unit(t) - t.radius + radius)
        # print 'final_target %d' % t.id
        # import pdb; pdb.set_trace()
        angle_to_target = me.get_angle_to_unit(t)
        have_time_to_turn_for_missile = HaveEnoughTimeToTurn(
            me, angle_to_target, t, state, action)
        if not have_time_to_turn_for_missile and not self.dodging:
             move.turn = angle_to_target

       
        if abs(angle_to_target) > abs(math.atan2(t.radius, distance)):
            move.action = ActionType.NONE
        if mes.fireball > 0 and me.mana < state.game.fireball_manacost:
            move.action = ActionType.NONE
        if mes.frost_bolt > 0 and me.mana < state.game.frost_bolt_manacost:
            move.action = ActionType.NONE
        
        
        if move.action == action:
            return
        if (HasMeleeTarget(me, state) and 
            (mes.staff_cooldown == 0)):
            move.action = ActionType.STAFF
            return
        if have_time_to_turn_for_missile:
            if distance < state.game.staff_range:
                melee_target = t
            else:
                melee_target = PickMeleeTarget(me, state)
            if melee_target is not None:
                angle_to_target = me.get_angle_to_unit(melee_target)
                if not self.dodging and not HaveEnoughTimeToTurn(
                    me, angle_to_target, melee_target, state, ActionType.STAFF):
                    move.turn = angle_to_target


class FleeInTerrorAction(MoveAction):
    def __init__(self, map_size, lane, safe_distance=30):
        MoveAction.__init__(self, map_size, lane)
        self.safe_distance = safe_distance

    @property
    def name(self):
        return "FLEE_IN_TERROR"

    def Act(self, me, state):
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        aggro = GetAggro(me, self.safe_distance, state)
        if aggro > 0:
            # print 'flee with aggro'
            if not self.dodging:
                self.MakeFleeMove(me, move, state)
        else:
            # print 'rush'
            if not self.dodging:
                self.MakeAdvanceMove(me, move, state)

        self.MaybeSetLanes(me, move)
        self.MaybeLearnSkills(me, move)  
        return move


class FleeAction(MoveAction):
    def __init__(self, map_size, lane, safe_distance=20, opt_range_allowance=20):
        MoveAction.__init__(self, map_size, lane)
        self.safe_distance = safe_distance
        self.opt_range_allowance = opt_range_allowance

    @property
    def name(self):
        return "FLEE"

    def Act(self, me, state):
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        aggro = GetAggro(me, self.safe_distance, state)
        target = None
        target = PickReachableTarget(me, me.cast_range, ActionType.MAGIC_MISSILE, state)
        if aggro > 0:
            # print 'flee with aggro'
            if not self.dodging:
                self.MakeFleeMove(me, move, state)
        elif ((target is not None) and 
              TargetInRangeWithAllowance(me, target, self.opt_range_allowance, state)):
            # print 'flee with target'
            if not self.dodging:
                self.MakeFleeMove(me, move, state)
        else:
            # print 'rush'
            if not self.dodging:
                self.MakeAdvanceMove(me, move, state)

        self.MakeMissileMove(me, move, state, target)
        self.MaybeSetLanes(me, move)
        self.MaybeLearnSkills(me, move)  
        return move


class AdvanceAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)

    @property
    def name(self):
        return "ADVANCE"

    def Act(self, me, state):
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        if not self.dodging:
            self.MakeAdvanceMove(me, move, state)
        self.MakeMissileMove(me, move, state)
        self.MaybeSetLanes(me, move)        
        self.MaybeLearnSkills(me, move)  
        return move


class RangedAttack(MoveAction):
    def __init__(self, map_size, lane, target, opt_range_allowance = 40):
        MoveAction.__init__(self, map_size, lane)
        self.target = target
        self.opt_range_allowance = opt_range_allowance

    @property
    def name(self):
        return "RANGE ATTACK %d" % self.target.id

    def Act(self, me, state):
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        if not TargetInRangeWithAllowance(me, self.target, self.opt_range_allowance):
            # import pdb; pdb.set_trace()
            if not self.dodging:
                self.RushToTarget(me, self.target, move, state)
        else:
            if not self.dodging:
                self.MakeFleeMove(me, move, state)
        self.MakeMissileMove(me, move, state, self.target)
        self.MaybeSetLanes(me, move)
        self.MaybeLearnSkills(me, move)
        return move


class FireballAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)

    @property
    def name(self):
        return "FIREBALL ATTACK"

    def Act(self, me, state):
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        t = state.my_state.fireball_target
        angle = me.get_angle_to_unit(t)
        if not self.dodging:
            self.RushToTarget(me, t, move, state)
        if abs(angle) < state.game.staff_sector / 2:
            move.action = ActionType.FIREBALL
            move.cast_angle = angle
            move.min_cast_distance = move.max_cast_distance = t.GetDistanceTo(me)
        elif not self.dodging:
            move.turn = angle
        state.dbg_line(me, t)
        speed = t - me
        if move.action == ActionType.FIREBALL:
            fake_p = Projectile(id=-me.id, x=me.x, y=me.y, speed_x=speed.x, speed_y=speed.y,
                angle=speed.GetAngle(), faction=me.faction, radius=state.game.fireball_radius, 
                type=ProjectileType.FIREBALL, owner_unit_id=me.id,
                owner_player_id=me.owner_player_id)
            from State import ProjectileState
            ps = ProjectileState(p=fake_p, me=me, game=state.game,
                                 world=state.world, last_state=state, dbg=state.dbg)
            if ps.end.GetSqDistanceTo(me) < t.GetSqDistanceTo(me) - EPSILON:
                move.action = ActionType.MAGIC_MISSILE
        self.MaybeSetLanes(me, move)      
        self.MaybeLearnSkills(me, move)  
        return move


class MeleeAttack(MoveAction):
    def __init__(self, map_size, lane, target, opt_range_allowance = 40):
        MoveAction.__init__(self, map_size, lane)
        self.target = target
        self.opt_range_allowance = opt_range_allowance

    @property
    def name(self):
        return "MELEE ATTACK %d" % self.target.id

    def Act(self, me, state):
        # if self.target.id == -21:
        #     import pdb; pdb.set_trace()
        move = Move()
        self.dodging = self.MaybeDodge(move, state)
        closest_target = GetClosestTarget(me, state)
        d = INFINITY
        if closest_target is not None:
            d = me.get_distance_to_unit(closest_target)
        if (not TargetInRangeWithAllowance(me, self.target, self.opt_range_allowance, state) or
            (isinstance(self.target, Wizard) and
             (not CanHitWizard(me, self.target, ActionType.MAGIC_MISSILE, state)))):
            if not self.dodging:
                self.RushToTarget(me, self.target, move, state)
        elif d > state.game.staff_range:
            if closest_target is None:
                if not self.dodging:
                    self.MakeAdvanceMove(me, move, state)
            else:
                if not self.dodging:
                    self.RushToTarget(me, closest_target, move, state)
        self.MakeMissileMove(me, move, state, self.target)
        self.MaybeSetLanes(me, move)      
        self.MaybeLearnSkills(me, move)  
        return move
