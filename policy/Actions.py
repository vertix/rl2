from model.ActionType import ActionType
from model.Game import Game
from model.LaneType import LaneType
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.Unit import Unit
from Geometry import Point
from Geometry import BuildPathAngle
from Geometry import BuildPath
from Analysis import GetAggro
from Analysis import PickTarget
from copy import deepcopy

import math


NUM_STEPS_PER_LANE = 3
WAYPOINT_RADIUS = 500
GRAPH_COOLDOWN = 20
MISSILE_DISTANCE_ERROR = 10


class NoOpAction(object):
    def Act(self, me, world, game):
        return Move()


def GetNextWaypoint(waypoints, me):
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


def MoveTowardsAngle(angle, move):
    move.speed = math.cos(angle) * 1000.
    move.strafe_speed = math.sin(angle) * 1000.


class MoveAction(object):
    def __init__(self, map_size, lane):
           self.lane = lane
           step = map_size / NUM_STEPS_PER_LANE
           start = Point(100, map_size - 100)
           end = Point(map_size - 100, 100)
           self.waypoints_by_lane = {
               LaneType.MIDDLE: [Point(i * step, map_size - i * step)
                                 for i in range(1, NUM_STEPS_PER_LANE - 1)],
               LaneType.TOP: [Point(100, map_size - i * step) for i in range(1, NUM_STEPS_PER_LANE - 1)] +
                             [Point(i * step, 100) for i in range(1, NUM_STEPS_PER_LANE - 1)],
               LaneType.BOTTOM: [Point(i * step, map_size - 100) for i in range(1, NUM_STEPS_PER_LANE - 1)] +
                                [Point(map_size - 100, map_size - i * step) for i in range(1, NUM_STEPS_PER_LANE - 1)]
           }
       
           for key in self.waypoints_by_lane:
               self.waypoints_by_lane[key] = [start] + self.waypoints_by_lane[key] + [end]
               
           self.last_graph_updated = -GRAPH_COOLDOWN

        # print self.waypoints_by_lane
        # import sys; sys.exit(10)
        
    def RushToTarget(self, me, target, move, game, world):
        # print me.x, me.y, target
        if self.last_graph_updated + GRAPH_COOLDOWN <= game.tick_count:
            self.path = BuildPath(me, target, game, world)
            self.last_graph_updated = game.tick_count
    
        if self.path is None: 
            angle = me.get_angle_to_unit(target)
        else:
            angle = BuildPathAngle(me, self.path)
        if angle is None:
            # TODO debug
            print 'WTF'
            angle = 0.3
        # print angle
        MoveTowardsAngle(angle, move)

        # print 'MV: (%.1f, %.1f) -> (%.1f, %.1f)' % (me.x, me.y, target[0], target[1])

        max_vector = [game.wizard_forward_speed, game.wizard_strafe_speed]
        optimal_angle = math.atan2(max_vector[1], max_vector[0])

        options = [angle - optimal_angle, angle + optimal_angle]
        target_angle = options[0] if abs(options[0]) < abs(options[1]) else options[1]
        move.turn = target_angle
        # print move.turn, move.speed, move.strafe_speed
        
    def MakeFleeMove(self, me, world, game, move):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetPrevWaypoint(waypoints, me)
        target = waypoints[i]
        self.RushToTarget(me, target, move, game, world)
        
    def MakeAdvanceMove(self, me, world, game, move):
        waypoints = self.waypoints_by_lane[self.lane]
        i = GetNextWaypoint(waypoints, me)
        target = waypoints[i]
        self.RushToTarget(me, target, move, game, world)
        
    def MakeMissileMove(self, me, world, game, move, target=None):
        t = deepcopy(target)
        if t is None:
            t = PickTarget(me, world, game)
        if t is None:
            return
        # print t.id, t.x, t.y
        # print t.faction, me.faction
        move.action = ActionType.MAGIC_MISSILE
        angle_to_target = me.get_angle_to_unit(t)
        distance = me.get_distance_to_unit(t)

        # print 'AT: (%.1f, %.1f) -> (%.1f, %.1f)' % (me.x, me.y, self.target.x, self.target.y)

        move.turn = angle_to_target
        if abs(angle_to_target) > abs(math.atan2(t.radius, distance)):
            move.action = ActionType.NONE

        
        move.min_cast_distance = me.get_distance_to_unit(t) - t.radius
        
        if distance > me.cast_range - t.radius + MISSILE_DISTANCE_ERROR:
            move.action = ActionType.NONE
        
        

class FleeAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)

    def Act(self, me, world, game):
        print 'flee'
        move = Move()
        aggro = GetAggro(me, game, world)
        print aggro
        if aggro > 0:
            self.MakeFleeMove(me, world, game, move)
        else:
            self.MakeAdvanceMove(me, world, game, move) 
        self.MakeMissileMove(me, world, game, move)
        return move


class AdvanceAction(MoveAction):
    def __init__(self, map_size, lane):
        MoveAction.__init__(self, map_size, lane)
    
        
    def Act(self, me, world, game):
        print 'advance'
        move = Move()
        self.MakeAdvanceMove(me, world, game, move)
        self.MakeMissileMove(me, world, game, move)
        return move


class RangedAttack(MoveAction):
    def __init__(self, map_size, lane, target):
        MoveAction.__init__(self, map_size, lane)
        self.target = target

    def Act(self, me, world, game):
        print 'ranged_attack'
        move = Move()
        self.MakeMissileMove(me, world, game, move, self.target)
            
        if (me.get_distance_to_unit(self.target) > 
            me.cast_range - self.target.radius + MISSILE_DISTANCE_ERROR):
            # import pdb; pdb.set_trace()
            self.RushToTarget(me, self.target, move, game, world)
        return move
