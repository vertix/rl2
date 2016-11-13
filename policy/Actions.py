from model.ActionType import ActionType
from model.Game import Game
from model.LaneType import LaneType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.Unit import Unit

import math


NUM_STEPS_PER_LANE = 40
WAYPOINT_RADIUS = 30

class NoOpAction(object):
    def Act(self, me, world, game):
        return Move()


def PointToUnit(point):
    return Unit(0, point[0], point[1], 0, 0, 0, 0)


def GetNextWaypoint(waypoints, me):
    last_point = waypoints[-1]
    for i, point in enumerate(waypoints):
        if me.get_distance_to(*point) < WAYPOINT_RADIUS:
            return min(i + 1, len(waypoints) - 1)
        if PointToUnit(point).get_distance_to(*last_point) < me.get_distance_to(*last_point):
            return i
    return len(waypoints) - 1


def GetPrevWaypoint(waypoints, me):
    first_point = waypoints[0]
    for i, point in reversed(list(enumerate(waypoints))):
        if i == 0:
            return 0
        if me.get_distance_to(*point) < WAYPOINT_RADIUS:
            return i-1
        if PointToUnit(point).get_distance_to(*first_point) < me.get_distance_to(*first_point):
            return i
    return 0


def MoveTowardsAngle(angle, move):
    move.speed = math.cos(angle) * 1000.
    move.strafe_speed = math.sin(angle) * 1000.


def RushToTarget(me, target, move, game):
    # TODO(vyakunin): try not strafing combination
    angle = me.get_angle_to(*target)
    MoveTowardsAngle(angle, move)

    # print 'MV: (%.1f, %.1f) -> (%.1f, %.1f)' % (me.x, me.y, target[0], target[1])

    max_vector = [game.wizard_forward_speed, game.wizard_strafe_speed]
    optimal_angle = math.atan2(max_vector[1], max_vector[0])

    options = [angle - optimal_angle, angle + optimal_angle]
    target_angle = options[0] if abs(options[0]) < abs(options[1]) else options[1]
    move.turn = target_angle


class MoveAction(object):
    def __init__(self, map_size, lane):
        self.lane = lane
        step = map_size / NUM_STEPS_PER_LANE
        self.waypoints_by_lane = {
            LaneType.MIDDLE: [(step, map_size - step), (4 * step, map_size - step),
                              (8 * step, map_size - step)] +
                             [(i * step, map_size - i * step)
                              for i in range(9, NUM_STEPS_PER_LANE)],
            LaneType.TOP: [(step, map_size - step)] +
                          [(2 * step, map_size - i * step) for i in range(5, NUM_STEPS_PER_LANE - 2)] +
                          [(i * step, 2 * step) for i in range(5, NUM_STEPS_PER_LANE - 2)],
            LaneType.BOTTOM: [(step, map_size - step)] +
                             [(i * step, map_size - 2 * step) for i in range(5, NUM_STEPS_PER_LANE - 2)] +
                             [(map_size - 2 * step, map_size - i * step) for i in range(5, NUM_STEPS_PER_LANE - 2)]
        }

        # print self.waypoints_by_lane
        # import sys; sys.exit(10)


class FleeAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)

    def Act(self, me, world, game):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetPrevWaypoint(waypoints, me)
        target = waypoints[i]
        move = Move()
        RushToTarget(me, target, move, game)
        return move


class AdvanceAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)
    
        
    def Act(self, me, world, game):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetNextWaypoint(waypoints, me)
        target = waypoints[i]
        move = Move()
        RushToTarget(me, target, move, game)
        return move


MISSILE_DISTANCE_ERROR = 10
class RangedAttack(object):
    def __init__(self, target):
        self.target = target

    def Act(self, me, world, game):
        move = Move()
        move.action = ActionType.MAGIC_MISSILE
        angle_to_target = me.get_angle_to_unit(self.target)
        distance = me.get_distance_to_unit(self.target)

        # print 'AT: (%.1f, %.1f) -> (%.1f, %.1f)' % (me.x, me.y, self.target.x, self.target.y)

        move.turn = angle_to_target
        if abs(angle_to_target) > abs(math.atan2(self.target.radius, distance)):
            move.action = ActionType.NONE

        if distance > me.cast_range - self.target.radius + MISSILE_DISTANCE_ERROR:
            MoveTowardsAngle(angle_to_target, move)
            move.action = ActionType.NONE

        return move
        # TODO(vyakunin): set min_cast_distance