from model.ActionType import ActionType
from model.Game import Game
from model.LaneType import LaneType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.Unit import Unit
from Geometry import Point
from Geometry import HasMeleeTarget
from Geometry import TargetInRangeWithAllowance
from CachedGeometry import Cache
from Analysis import GetAggro
from Analysis import PickTarget
from Analysis import PickReachableTarget
from Analysis import GetRemainingActionCooldown
from Analysis import HaveEnoughTimeToTurn
from Analysis import GetMaxForwardSpeed
from Analysis import GetMaxStrafeSpeed
from Analysis import FindUnitById
from copy import deepcopy

import math


NUM_STEPS_PER_LANE = 6
WAYPOINT_RADIUS = 300 # must be less than distance between waypoints!
TARGET_COOLDOWN = 20
MISSILE_DISTANCE_ERROR = 10
MARGIN = 200
last_x = -1
last_y = -1


class NoOpAction(object):
    def Act(self, me, world, game):
        return Move()


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


def MoveTowardsAngle(me, game, angle, move, d):
    if abs(angle) < math.pi / 2:
        move.speed = math.cos(angle) * GetMaxForwardSpeed(me, game) * 100
    else:
        move.speed = math.cos(angle) * GetMaxStrafeSpeed(me, game) * 100
    move.strafe_speed = math.sin(angle) * GetMaxStrafeSpeed(me, game) * 100
    vd = math.hypot(move.speed, move.strafe_speed)
    if vd > d:
        move.speed /= vd / d
        move.strafe_speed /= vd / d


class MoveAction(object):
    def __init__(self, map_size, lane):
        self.lane = lane
        step = map_size / NUM_STEPS_PER_LANE
        start = Point(MARGIN, map_size - MARGIN)
        end = Point(map_size - MARGIN, MARGIN)
        self.waypoints_by_lane = {
            LaneType.MIDDLE: [Point(i * step, map_size - i * step)
                              for i in range(1, NUM_STEPS_PER_LANE - 1)],
            LaneType.TOP: [Point(MARGIN, map_size - i * step)
                           for i in range(1, NUM_STEPS_PER_LANE - 1)] +
                          [Point(i * step, MARGIN) for i in range(1, NUM_STEPS_PER_LANE - 1)],
            LaneType.BOTTOM: [Point(i * step, map_size - MARGIN)
                              for i in range(1, NUM_STEPS_PER_LANE - 1)] +
                             [Point(map_size - MARGIN, map_size - i * step)
                              for i in range(1, NUM_STEPS_PER_LANE - 1)]
        }

        for key in self.waypoints_by_lane:
            self.waypoints_by_lane[key] = [start] + self.waypoints_by_lane[key] + [end]
        self.focus_target = None

    def RushToTarget(self, me, target, move, game, world):
        path = Cache.GetInstance().GetPathToTarget(me, target, game, world)
        t_id = -1
        angle = 0
        d = 10
        if path is None:
            angle = me.get_angle_to_unit(target)
        else:
            angle, d, t_id = path.GetNextAngleDistanceAndTarget(me)
        if angle is None:
            angle = me.get_angle_to_unit(target)
        MoveTowardsAngle(me, game, angle, move, d)
        self.overridden_target = None
        if t_id != -1:
            new_target = FindUnitById(world, t_id)
            if new_target is not None:
                self.overridden_target = new_target
                return

        max_vector = [GetMaxForwardSpeed(me, game), GetMaxStrafeSpeed(me, game)]
        optimal_angle = math.atan2(max_vector[1], max_vector[0])

        options = [angle - optimal_angle, angle + optimal_angle]
        target_angle = options[0] if abs(options[0]) < abs(options[1]) else options[1]
        move.turn = target_angle
        return None

    def MakeFleeMove(self, me, world, game, move):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetPrevWaypoint(waypoints, me)
        target = waypoints[i]
        # print Point.FromUnit(me), target
        self.RushToTarget(me, target, move, game, world)
            

    def MakeAdvanceMove(self, me, world, game, move):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetNextWaypoint(waypoints, me)
        target = waypoints[i]
        self.RushToTarget(me, target, move, game, world)

    def MakeMissileMove(self, me, world, game, move, target=None):
        t = (deepcopy(self.overridden_target) if self.overridden_target is not None 
             else deepcopy(target))
        if t is None:
            if (self.focus_target != None and
                    world.tick_index < self.last_target + TARGET_COOLDOWN):
                t = self.focus_target
            else:
                t = PickTarget(me, world, game)
                self.focus_target = t
                self.last_target = world.tick_index
        if t is None:
            return
        distance = me.get_distance_to_unit(t)
        angle_to_target = me.get_angle_to_unit(t)
        if not HaveEnoughTimeToTurn(me, angle_to_target, t, game):
             move.turn = angle_to_target
        if GetRemainingActionCooldown(me) == 0:
            move.action = ActionType.MAGIC_MISSILE

       
        if abs(angle_to_target) > abs(math.atan2(t.radius, distance)):
            move.action = ActionType.NONE

        move.min_cast_distance = min(
            me.cast_range, 
            me.get_distance_to_unit(t) - t.radius + game.magic_missile_radius)
        if not TargetInRangeWithAllowance(me, t, game.magic_missile_radius):
            move.action = ActionType.NONE
        
        if move.action == ActionType.MAGIC_MISSILE:
            return
        if HasMeleeTarget(me, world, game):
            move.action = ActionType.STAFF


class FleeAction(MoveAction):
    def __init__(self, map_size, lane, safe_distance=20, opt_range_allowance=20):
        MoveAction.__init__(self, map_size, lane)
        self.safe_distance = safe_distance
        self.opt_range_allowance = opt_range_allowance

    def Act(self, me, world, game):
        move = Move()
        aggro = GetAggro(me, game, world, self.safe_distance)
        target = None
        target = PickReachableTarget(me, world, game)
        if aggro > 0:
            # print 'flee with aggro'
            # import pdb; pdb.set_trace()
            self.MakeFleeMove(me, world, game, move)
        elif ((target is not None) and 
              TargetInRangeWithAllowance(me, target, self.opt_range_allowance)):
            # print 'flee with target'
            self.MakeFleeMove(me, world, game, move)
        else:
            # print 'rush'
            self.MakeAdvanceMove(me, world, game, move)

        self.MakeMissileMove(me, world, game, move, target)
        return move


class AdvanceAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)

    def Act(self, me, world, game):
        move = Move()
        self.MakeAdvanceMove(me, world, game, move)
        self.MakeMissileMove(me, world, game, move)
        return move


class RangedAttack(MoveAction):
    def __init__(self, map_size, lane, target, opt_range_allowance = 40):
        MoveAction.__init__(self, map_size, lane)
        self.target = target
        self.opt_range_allowance = opt_range_allowance

    def Act(self, me, world, game):
        move = Move()
        
        if not TargetInRangeWithAllowance(me, self.target, self.opt_range_allowance):
            # import pdb; pdb.set_trace()
            self.RushToTarget(me, self.target, move, game, world)
        else:
            self.MakeFleeMove(me, world, game, move)
        self.MakeMissileMove(me, world, game, move, self.target)
        return move
