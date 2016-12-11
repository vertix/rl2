from collections import namedtuple
from math import pi
from math import hypot
from math import acos
from math import asin
from math import cos
from math import sin
from math import atan2
from math import sqrt
from copy import deepcopy
from Graph import Dijkstra
from model.CircularUnit import CircularUnit
from model.LivingUnit import LivingUnit
from model.Wizard import Wizard
from model.Minion import Minion
from model.MinionType import MinionType
from model.LaneType import LaneType
from model.Tree import Tree
from Colors import RED
from Colors import GREEN

EPSILON = 1E-4
MACRO_EPSILON = 2
MAX_OBSTACLES = 8
MAX_TREES = 3
INFINITY = 1e6
TICKS_TO_ACCOUNT_FOR = 10
RADIUS_ALLOWANCE = 2
HALF_LANE_WIDTH = 500

def GetLanes(u):
    lanes = []
    if u is None:
        return lanes
    p = Point.FromUnit(u)
    bottom_left = Point(300, 3700)
    bottom_right = Point(3700, 3700)
    top_left = Point(300, 300)
    top_right = Point(3700, 300)
    left = Line(bottom_left, top_left)
    right = Line(bottom_right, top_right)
    top = Line(top_right, top_left)
    bottom = Line(bottom_left, bottom_right)
    diagonal = Line(bottom_left, top_right)

    if ((p.GetDistanceToLine(left) < HALF_LANE_WIDTH) or 
        (p.GetDistanceToLine(top) < HALF_LANE_WIDTH)):
        lanes.append(LaneType.TOP)
    if ((p.GetDistanceToLine(right) < HALF_LANE_WIDTH) or 
        (p.GetDistanceToLine(bottom) < HALF_LANE_WIDTH)):
        lanes.append(LaneType.BOTTOM)
    if p.GetDistanceToLine(diagonal) < HALF_LANE_WIDTH: 
        lanes.append(LaneType.MIDDLE)
    return lanes

class Transition(object):
    def __init__(self, begin, end, edge):
        self.begin = begin
        self.end = end
        self.edge = edge
        self.circle = edge.circle
        self.type = edge.type
        if self.type == Edge.ARC:
            # import pdb; pdb.set_trace()
            self.begin_angle = (self.begin - self.circle).GetAngle()
            self.end_angle = (self.end - self.circle).GetAngle()   
            
    def Show(self, state):
        color = GREEN
        if self.type == Edge.ARC:
            color = RED
        state.dbg_line(self.begin, self.end, color)
        for id in self.edge.target_ids:
            if id in state.index:
                state.dbg_line(self.begin, state.index[id].unit)
            
    def GetTargetIds(self):
        return self.edge.target_ids
        
    def IsNonTrivial(self):
        if self.type == Edge.ARC and self.circle.radius < EPSILON:
            return False
        if self.begin.GetDistanceTo(self.end) < EPSILON:
            return False
        return True
        
    def GetDistanceTo(self, p):
        a = Point.FromUnit(p)
        if self.type == Edge.SEGMENT:
            return a.GetDistanceToSegment(Segment(self.begin, self.end))
        # import pdb; pdb.set_trace()
        angle = (a - self.circle).GetAngle()
        if (abs(angle - self.begin_angle) + abs(angle - self.end_angle) - 
            abs(self.begin_angle - self.end_angle)) < EPSILON:
            return max(0.0, a.GetDistanceTo(self.circle) - self.circle.radius)
        return min(a.GetDistanceTo(self.begin), a.GetDistanceTo(self.end))        
        
    def GetAngleAndDistanceFrom(self, u):
        if self.type != Edge.ARC:
            return (u.get_angle_to_unit(self.end), u.get_distance_to_unit(self.end))
        v_to_c = Point.FromUnit(self.circle) - u
        v_to_c_angle = v_to_c.GetAngle()
        d = u.get_distance_to_unit(self.circle)
        # print u, self.begin, self.end, self.circle
        # print self.circle.radius / d
        delta = asin(min(1.0, self.circle.radius / d))
        if self.circle.radius > d + EPSILON:
            delta += pi/12
            
        if self.begin_angle < self.end_angle:
            # positive turn, therefore I should turn right
            angle = v_to_c_angle - delta
        else:
            angle = v_to_c_angle + delta
        return (NormAngle(angle - u.angle), 10)
        
class PlainCircle(object):
    def __init__(self, p, r):
        self.x = p.x
        self.y = p.y
        self.p = Point.FromUnit(p)
        self.radius = r
        self.faction = None

    def get_distance_to_unit(self, u):
        return self.p.GetDistanceTo(u)

class Path(object):
    # waypoints = [(point, edge)]
    def __init__(self, waypoints):
        self.transitions = []
        for i, w in enumerate(waypoints[:-1]):
            t = Transition(w[0], waypoints[i+1][0], waypoints[i][1])
            if t.IsNonTrivial():
                self.transitions.append(t)
    
    def GetCurrentTransition(self, u):
        current = None
        closest = INFINITY
        for t in reversed(self.transitions):
            d = t.GetDistanceTo(u)
            if d < closest - MACRO_EPSILON:
                closest = d
                current = t
        return current
    
    def GetNextAngleDistanceAndTargets(self, u):
        current = self.GetCurrentTransition(u)
        if current is None:
            return None
        return current.GetAngleAndDistanceFrom(u) + (current.GetTargetIds(),)
    
    def Show(self, state):
        for t in self.transitions:
            t.Show(state)
        

class Line(object):
    def __init__(self, p1, p2):
        self.a = p1.y - p2.y
        self.b = p2.x - p1.x
        self.c = -self.a * p1.x - self.b * p1.y
        self.sq_norm = self.a * self.a + self.b * self.b
    
    def Normal(self):
        return Point(self.a, self.b) * (1.0 / sqrt(self.sq_norm))
    
    def IntersectWithCircle(self, c):
        # Ax + By + C = 0
        # (x - c.x) ^ 2 + (y - c.y) ^ 2 = c.r ^ 2
        if c.p.GetDistanceToLine(self) > c.radius - EPSILON:
            return []
        a = acos(c.p.GetDistanceToLine(self) / c.radius)
        return [self.Normal().Rotate(a) * c.radius + c, self.Normal().Rotate(-a) * c.radius + c]
        
    def __str__(self):
        return '(%.6f*x + %.6f*y + %.6f = 0)' % (self.a, self.b, self.c)
    
    def __repr__(self):
        return self.__str__()

class Segment(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.l = Line(p1, p2)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @classmethod
    def FromUnit(cls, u):
        return cls(u.x, u.y)
    
    def ProjectToLine(self, l, additional_shift=0):
        #     ax + by + c = 0
        #     x = x0 + at
        #     y = y0 + bt
        #     ax0 + aat + by0 + bbt + c = 0
        #     t = (-ax0 - by0 - c) / (aa + bb)
        t = (-l.a * self.x - l.b + self.y - l.c) / l.sq_norm
        if t < 0:
            t -= additional_shift / l.sq_norm
        else:
            t += additional_shift / l.sq_norm
        return Point(self.x + t * l.a, self.y + t * l.b)

    def Rotate(self, a):
        return Point(self.x * cos(a) - self.y * sin(a), self.x * sin(a) + self.y * cos(a))
        
    def GetDistanceTo(self, p):
        return (self - p).Norm()

    def GetSqDistanceTo(self, p):
        return (self - p).SqNorm()
        
    def ScalarMul(self, p):
        return self.x * p.x + self.y * p.y
    
    def Norm(self):
        return hypot(self.x, self.y)
        
    def SqNorm(self):
        return self.x * self.x + self.y * self.y
        
    def GetDistanceToLine(self, l):
        return abs(self.x * l.a + self.y * l.b + l.c) / hypot(l.a, l.b)

    def GetSqDistanceToLine(self, l):
        return ((self.x * l.a + self.y * l.b + l.c) * 
                (self.x * l.a + self.y * l.b + l.c) / l.sq_norm)
        
    def GetDistanceToSegment(self, s):
        p1 = s.p1
        p2 = s.p2
        ans = min(self.GetDistanceTo(p1), self.GetDistanceTo(p2))
        if ((self - p1).ScalarMul(p2 - p1) > 0) and ((self - p2).ScalarMul(p1 - p2) > 0):
            ans = min(ans, self.GetDistanceToLine(Line(p1, p2)))
            
        return ans

    def GetSqDistanceToSegment(self, s):
        p1 = s.p1
        p2 = s.p2
        ans = min(self.GetSqDistanceTo(p1), self.GetSqDistanceTo(p2))
        if ((self - p1).ScalarMul(p2 - p1) > 0) and ((self - p2).ScalarMul(p1 - p2) > 0):
            ans = min(ans, self.GetSqDistanceToLine(s.l))
            
        return ans
    
    def GetAngle(self):
        return atan2(self.y, self.x)
        
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __mul__(self, f):
        return Point(self.x * f, self.y * f)
        
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __str__(self):
        return 'Point(%.3f, %.3f)' % (self.x, self.y)
    
    def __repr__(self):
        return self.__str__()
    

def BuildObstacles(me, state):
   obstacles = [(me.get_distance_to_unit(unit), deepcopy(unit)) for unit in 
       state.world.wizards + state.world.minions + state.world.trees + state.world.buildings 
       if me.get_distance_to_unit(unit) < me.vision_range and me.id != unit.id]
   obstacles.sort()
   obstacles = [Obstacle(me, o) for _, o in obstacles]
   new_obstacles = []
   total = 0
   trees = 0
   for o in obstacles:
       if total == MAX_OBSTACLES:
           break
       if o.is_tree:
           if trees < MAX_TREES:
               trees += 1
               total += 1
               new_obstacles.append(o)
       else:
           total += 1
           new_obstacles.append(o)
   return new_obstacles


def BuildTangentPoint(u, v, alpha):
    v_rotated = v.Rotate(alpha)
    return (v_rotated + u, v_rotated.GetAngle())


# u1, u2: CircularUnit
# returns: list((point1, alpha1, point2, alpha2)) of tangents with pointX on
# uX having angle alphaX.(-pi <= alphaX < pi)
def BuildTangents(u1, u2):
    r1 = u1.radius
    r2 = u2.radius
    if r1 > r2:
        return [(p2, a2, p1, a1) for (p1, a1, p2, a2) in BuildTangents(u2, u1)]
    # now r1 < r2
    u1p = Point.FromUnit(u1)
    u2p = Point.FromUnit(u2)
    d = u1.get_distance_to_unit(u2)
    if abs(r2) < EPSILON:
        # both circles are points
        return [(u1p, 0, u2p, 0)]
    if r2 > d + r1 - EPSILON:
        # they are inside each other
        return []
    result = []
    # vector from center1 to point on circle1 closest to circle2
    v1 = (u2p - u1p) * (r1 / d)
    # same but for circle2
    v2 = (u1p - u2p) * (r2 / d)
    # calculating outer tangents
    alpha2 = acos((r2 - r1) / d)
    alpha1 = pi - alpha2
    result.append(BuildTangentPoint(u1, v1, alpha1) + BuildTangentPoint(u2, v2, -alpha2))
    result.append(BuildTangentPoint(u1, v1, -alpha1) + BuildTangentPoint(u2, v2, alpha2))
    if d < r1 + r2 + EPSILON:
        # they intersect
        return result
    if abs(r1) < EPSILON:
        # outer and inner are the same
        return result
    # calculating inner tangents
    alpha = acos((r1 + r2) / d)
    result.append(BuildTangentPoint(u1, v1, alpha) + BuildTangentPoint(u2, v2, alpha))
    result.append(BuildTangentPoint(u1, v1, -alpha) + BuildTangentPoint(u2, v2, -alpha))
    return result
    
def SegmentCrossesCircle(s, o):
    if (o.radius < EPSILON):
        return False
    op = Point.FromUnit(o)
    return op.GetSqDistanceToSegment(s) - o.radius * o.radius < -EPSILON

def SegmentClearFromHardObstacles(s, obstacles, soft_obstacles, state):
    obs = []
    for i, o in enumerate(obstacles):
        if SegmentCrossesCircle(s, o):
            if o.is_tree:
                obs.append((o.id, o))
            else:
                return False
    obs.sort(key=lambda x: s.p1.GetSqDistanceTo(x[1]))
    for o in obs:
        soft_obstacles.append(o[0])
    return True
    
# returns ([p1, p2], [(a1, a2), (b1, b2)]) or None, where pX - points, aX - corresponding
# angles on c1, bX - corresponding angles on c2 intersection goes from a1 to a2 in positive
# direction on c1 and in negative on c2.
def IntersectCircles(c1, c2):
    d = c1.get_distance_to_unit(c2)
    if c1.radius + c2.radius < d + EPSILON:
        return None
    if d < abs(c1.radius - c2.radius) + EPSILON:
        return None
    if abs(d) < EPSILON:
        return None 
    c1p = Point.FromUnit(c1)
    c2p = Point.FromUnit(c2)

    a = (c1.radius * c1.radius - c2.radius * c2.radius + d * d) / (2 * d)
    p0 = c1p + (c2p - c1p) * (a / d)
    h = sqrt(c1.radius * c1.radius - a * a)
    delta = (c2p - c1p).Rotate(pi/2) * (h / d)
    p1 = p0 - delta
    p2 = p0 + delta
        
    return ([p1, p2], [((p1 - c1p).GetAngle(), (p2 - c1p).GetAngle()), 
                       ((p1 - c2p).GetAngle(), (p2 - c2p).GetAngle())])

def IntersectSegments(s1, s2):
    if s1.l.sq_norm < EPSILON:
        return None
    if s2.l.sq_norm < EPSILON:
        return None
    # A1x + B1y + C1 = 0
    # A2x + B2y + C2 = 0
    # d = A1B2 - A2B1
    # x = (-C1 * B2 + C2 * B1) / d
    # y = (A1 * -C2 + A2 * C1) / d
    d = s1.l.a * s2.l.b - s2.l.a * s1.l.b
    if abs(d) < EPSILON:
        return None
    ans = Point((s2.l.c * s1.l.b - s1.l.c * s2.l.b) / d,
                (s1.l.c * s2.l.a - s2.l.c * s1.l.a) / d)
    if ((s1.p2 - s1.p1).ScalarMul(ans - s1.p1) > EPSILON and
        (s1.p1 - s1.p2).ScalarMul(ans - s1.p2) > EPSILON and
        (s2.p2 - s2.p1).ScalarMul(ans - s2.p1) > EPSILON and
        (s2.p1 - s2.p2).ScalarMul(ans - s2.p2) > EPSILON):
        return ans
    return None
                       
# a1, a2 - starting and ending angles for arc, i1 and i2 - corresponding point numbers,
# returns [(-pi <= alpha < pi, type, point_no)...],
# type is 1 for beginning of intersection and -1 for end of intersection. point_no = -1 
# means fake point.
def AddArc(a1, a2, i1, i2):
    if a1 > a2 + EPSILON:
        # break arc into two to wrap pi point, assign fake point ID.
        return [(a1, 1, i1), (pi, -1, -1), (-pi, 1, -1), (a2, -1, i2)]
    return [(a1, 1, i1), (a2, -1, i2)]
    
def Invalid(p, state):
    r = state.world.wizards[0].radius + MACRO_EPSILON
    return ((p.x < r) or (p.x > state.world.width - r) or
            (p.y < r) or (p.y > state.world.height - r))
    
class Edge(object):
    SEGMENT = 0
    ARC = 1
    def __init__(self, v, w, type, circle, target_ids=[]):
        self.v = v
        self.w = w
        self.type = type
        self.circle = circle
        self.target_ids = target_ids

def GetDamagePerTicks(damage, remaining_cooldown, cooldown, ticks):
    return (max(0, int(ticks - remaining_cooldown + cooldown - 1)) /
            int(cooldown) * damage)

def GetTreeCost(start, ids, state):
    if not ids:
        return 0.0
    ms = state.my_state
    cd = 0
    scd = 0
    speed = ms.max_speed
    d = start.GetDistanceTo(state.index[ids[0]].unit)
    ret = 0.0
    for i, id in enumerate(ids):
        tree = state.index[id].unit
        hp = tree.life
        if i > 0:
            d += tree.get_distance_to_unit(state.index[ids[i-1]].unit)
        new_d = min(d, ms.attack_range + tree.radius)
        ticks = (d - new_d) / speed
        cd -= ticks
        scd = max(0, scd - ticks)
        
        d = new_d
        
        new_d = ms.unit.radius + tree.radius
        ticks = (d - new_d) / speed
        
        hp -= GetDamagePerTicks(ms.missile, cd, ms.missile_total_cooldown, ticks)
        cd -= ticks
        scd -= ticks
        while cd < 0:
            cd += ms.missile_total_cooldown
        while scd < 0:
            scd += state.game.staff_cooldown_ticks
        d = new_d
        
        while hp > 0:
            if cd < scd:
                ret += cd
                scd -= cd
                hp -= ms.missile
                cd = ms.missile_total_cooldown
            else:
                ret += scd
                cd -= scd
                hp -= ms.staff
                scd = state.game.staff_cooldown_ticks
    return ret * speed

def AddEdge(points, i1, i2, d, g, state, type=Edge.SEGMENT, circle=None, target_ids=[]):
    if Invalid(points[i1], state) or Invalid(points[i2], state):
        return
    g[i1].append(Edge(i2, d + GetTreeCost(points[i1], target_ids, state),
                 type, circle, target_ids))
    g[i2].append(Edge(i1, d + GetTreeCost(points[i2], list(reversed(target_ids)), state),
                 type, circle, list(reversed(target_ids))))

def GetAngleDiff(a1, a2):
    alpha = abs(a1 - a2)
    while alpha > 2 * pi:
        alpha -= 2 * pi
    return alpha

def GetArcLength(a1, a2, r):
    return GetAngleDiff(a1, a2) * r

def Downcast(ancestor, descendent):
    """
        automatic downcast conversion.....

        (NOTE - not type-safe -
        if ancestor isn't a super class of descendent, it may well break)

    """
    for name, value in vars(ancestor).iteritems():
        setattr(descendent, name, value)

    return descendent
    
class Obstacle(CircularUnit):
    def __init__(self, me, u):
        CircularUnit.__init__(self, u.id, u.x, u.y, u.speed_x, u.speed_y,
                              u.angle, u.faction, u.radius)
        self.is_tree = isinstance(u, Tree)
            
# target = (x, y)
# units = list(CircularUnit)
# TODO(vyakunin): decide on a better interface
# returns [(point, previous_point_no, shortest_distance)], points_per_unit: list(list(int)))
def FindOptimalPaths(me, units, state):
    me_point = Obstacle(me, me)
    me_point.radius = 0
    all_units = deepcopy(units)
    for o in units:
        if abs(o.speed_x) + abs(o.speed_y) > EPSILON:
            new_o = deepcopy(o)
            new_o.x += o.speed_x * TICKS_TO_ACCOUNT_FOR
            new_o.y += o.speed_y * TICKS_TO_ACCOUNT_FOR
            all_units.append(new_o)
    for o in all_units:  
        o.radius = o.radius + me.radius + RADIUS_ALLOWANCE
        if me.get_distance_to_unit(o) < o.radius + EPSILON:
            o.radius = max(0, me.get_distance_to_unit(o) - EPSILON)
    all_units = [me_point] + all_units
    
    for u in all_units:
        state.dbg_circle(u)
    
    # points[i]: Point
    points = []
    # graph[point_no_i]: [(point_no_j, direct_distance_i_to_j), ...]}
    graph = []

    # points_per_unit[unit_no_i]: [(-pi <= alpha < pi, type, point_no)...],
    # type is 1 for beginning of intersection, 0 for regular point and -1 for end of 
    # intersection.
    points_per_unit = []
    for i in range(len(all_units)):
        points_per_unit.append([])
    for i, u1 in enumerate(all_units):
        for j, u2 in enumerate(all_units[i+1:]):
            j += i + 1
            tangents = BuildTangents(u1, u2)
            for p1, a1, p2, a2 in tangents:
                soft_obstacles_ids = []
                if SegmentClearFromHardObstacles(Segment(p1, p2), all_units,
                                                 soft_obstacles_ids, state):
                    i1 = len(points)
                    i2 = i1 + 1
                    points.append(p1)
                    points.append(p2)
                    graph.append([])
                    graph.append([])
                    points_per_unit[i].append((a1, 0, i1))
                    points_per_unit[j].append((a2, 0, i2))
                    distance = p1.GetDistanceTo(p2)
                    AddEdge(points, i1, i2, distance, graph, state,
                            Edge.SEGMENT, None, soft_obstacles_ids)

            intersections = IntersectCircles(u1, u2)
            if intersections is not None:
                p, a = intersections
                i1 = len(points)
                i2 = i1 + 1
                points.extend(p)
                graph.append([])
                graph.append([])
                points_per_unit[i].extend(AddArc(a[0][0], a[0][1], i1, i2))
                points_per_unit[j].extend(AddArc(a[1][1], a[1][0], i2, i1))
            
            
    for unit_index, p in enumerate(points_per_unit):
        u = all_units[unit_index]
        p.sort()
        for i in range(len(p)):
            # only add arc edges not passing intersection arcs. Keep those
            # inside intersections isolated from those outside, so they are unreachable.
            if p[i-1][1] == 0 and p[i][1] == 0:
                AddEdge(points, p[i-1][2], p[i][2], 
                        GetArcLength(p[i-1][0], p[i][0], all_units[unit_index].radius),
                        graph, state, type=Edge.ARC, circle=u)
        # leave just list of point numbers
        points_per_unit[unit_index] = [k for _, __, k in p if k != -1]
    
    optimal_distances, prev = Dijkstra(graph)
    return (points, prev, optimal_distances, points_per_unit)
    

# returns [[point, is_arc, circle]]
def BuildPath(me, target, state):
    obstacles = BuildObstacles(me, state)
    if isinstance(target, LivingUnit):
        t = deepcopy(target)
    else:
        t = LivingUnit(0, target.x, target.y, 0, 0, 0, 0, 0, 0, 0, [])
    obstacles = [o for o in obstacles if o.get_distance_to_unit(target) > 
                                         o.radius - t.radius]
    p, prev, d, points_per_unit = FindOptimalPaths(
        me, [Obstacle(me, t)] + obstacles, state)
    t_id = -1
    min_d = INFINITY
    for ps in points_per_unit[1]:
        if t_id == -1 or d[ps] < min_d:
            min_d = d[ps]
            t_id = ps
    if t_id == -1:
        # import pdb; pdb.set_trace()
        # print 'no way to ' + str(target)
        return None
    # import pdb; pdb.set_trace()
    path = [(p[t_id], prev[t_id][1])]
    
    while prev[t_id][0] != -1:
        t_id = prev[t_id][0]
        path.append((p[t_id], prev[t_id][1]))
    path.reverse()
    
    for i, p in enumerate(path[:-1]):
        path[i] = (path[i][0], path[i+1][1])
    return Path(path)
    
def SegmentsIntersect(l1, r1, l2, r2):
    if abs(l1 - l2) + abs(r1 - r2) < EPSILON:
        return True
    return (((l1 < l2 - EPSILON) and (r1 > l2 + EPSILON))
            or ((l2 < l1 - EPSILON) and (r2 > l1 + EPSILON)))

def NormAngle(a):
    res = a
    while res < -pi - EPSILON:
        res += 2 * pi
    while res > pi - EPSILON:
        res -= 2 * pi
    return res

def ArcsIntersect(l1, r1, l2, r2):
    l1 = NormAngle(l1)
    l2 = NormAngle(l2)
    r1 = NormAngle(r1)
    r2 = NormAngle(r2)
    a1 = [(l1, r1)]
    if l1 > r1:
        a1 = [(l1, pi), (pi, r2)]
    a2 = [(l2, r2)]
    if l2 > r2:
        a2 = [(l2, pi), (pi, r2)]
    for a in a1:
        for b in a2:
            if SegmentsIntersect(a[0], a[1], b[0], b[1]):
                return True
    return False
    
def SectorCovers(u, a, r, t):
    return ((u.get_distance_to_unit(t) < r + t.radius - EPSILON) and 
            (abs(u.get_angle_to_unit(t)) < a / 2.0 - EPSILON))
    
def HasMeleeTarget(u, state):
    a = None
    r = None
    if isinstance(u, Wizard):
        a = state.game.staff_sector
        r = state.game.staff_range
    if isinstance(u, Minion) and (u.type == MinionType.ORC_WOODCUTTER):
        a = orc_woodcutter_attack_sector
        r = orc_woodcutter_attack_range
    if r is None:
        return False
    for t in (state.world.wizards + state.world.minions +
              state.world.buildings + state.world.trees):
        from Analysis import IsEnemy
        if (IsEnemy(u, t) or isinstance(t, Tree)) and SectorCovers(u, a, r, t):
            return True
    return False

def RangeAllowance(me, target, state):
    d = me.get_distance_to_unit(target) - target.radius
    ret = me.cast_range - d
    return ret

def TargetInRangeWithAllowance(me, target, allowance, state):
    return RangeAllowance(me, target, state) > allowance

def ProjectileWillHit(ps, t):
    if ps.unit.faction == t.faction:
        return False
    return (Point.FromUnit(t).GetSqDistanceToSegment(Segment(ps.start, ps.end)) <
        (t.radius + ps.min_radius) * (t.radius + ps.min_radius))
        
def GetMaxSpeedTowardsAngle(mes, angle):
    if angle > pi / 2:
        return mes.strafe_speed
    # hypot(strafe_speed / max_strafe_speed, speed / max_speed) = 1
    speed = Point(1, 0).Rotate(angle)
    return 1 / hypot(speed.x / mes.forward_speed, speed.y / mes.strafe_speed)

class DirectionAndTime(object):
    def __init__(self, mes, direction):
        self.direction = direction
        me = mes.unit
        angle = abs(me.get_angle_to_unit(direction))
        d = me.get_distance_to_unit(direction)
        sum_time = 0.0
        if angle > pi/2 + EPSILON:
            time = (angle - pi/2) / mes.max_rotation_speed
            max_reduction = time * mes.strafe_speed
            if d < max_reduction + EPSILON:
                self.time = d / mes.strafe_speed
                return
            d -= max_reduction
            sum_time += time
        if angle > EPSILON:
            time = angle / mes.max_rotation_speed
            # linear interpolation is a simplification
            current_speed = GetMaxSpeedTowardsAngle(mes, angle)
            max_reduction = (current_speed + mes.forward_speed) / 2.0 * time
            if d < max_reduction + EPSILON:
                # simplify to linear accelerated motion
                # a = (mes.forward_speed - current_speed) / time
                # d = a^2*x / 2
                # x = 2 * d / a^2
                a = (mes.forward_speed - current_speed) / time
                self.time = sum_time + 2 * d / a / a
                return
            d -= max_reduction
            sum_time += time
        sum_time += d / mes.forward_speed
        self.time = sum_time

def PickDodgeDirectionAndTime(me, ps, state, extra_space_required=MACRO_EPSILON):
    mes = state.index[me.id]
    psp = Point.FromUnit(ps.end)
    candidates = []
    mep = Point.FromUnit(me)
    v = mep - psp
    d = psp.GetDistanceTo(me)
    if d < EPSILON:
        v = psp - ps.start
    candidates.append(mep + v * ((ps.max_radius + me.radius - d + extra_space_required) / v.Norm()))
    candidates.append(mep.ProjectToLine(ps.border1.l, extra_space_required))
    candidates.append(mep.ProjectToLine(ps.border2.l, extra_space_required))
    obstacles = []
    for u in state.world.trees + state.world.wizards + state.world.buildings + state.world.minions:
        if psp.GetSqDistanceTo(u) < ps.max_radius + u.radius + me.radius + extra_space_required:
            obstacles.append(u)
            # ([p1, p2], [(a1, a2), (b1, b2)]) or None, where pX - points, aX - corresponding
            # angles on c1, bX - corresponding angles on c2 intersection goes from a1 to a2 in positive
            # direction on c1 and in negative on c2.
            intersections = IntersectCircles(
                PlainCircle(psp, ps.max_radius + me.radius + extra_space_required),
                PlainCircle(u, u.radius + me.radius))
            if intersections:
                candidates.extend(intersections[0])
        if ps.start.GetDistanceTo(u) < ps.max_radius + u.radius + me.radius + extra_space_required:
            obstacles.append(u)
    best = None
    for c in candidates:
        if (c.GetDistanceToSegment(ps.center_line) <
            ps.max_radius + me.radius + extra_space_required):
            continue
        good = True
        for o in obstacles:
            if SegmentCrossesCircle(Segment(mep, c), o):
                good = False
                break
        if good:
            dt = DirectionAndTime(mes, c)
            if best is None or dt.time < best.time:
                best = dt
    if best is not None:
        projectile_distance = min(ps.end.GetDistanceTo(ps.start),
                                  best.direction.ProjectToLine(ps.center_line.l).GetDistanceTo(
                                      ps.start))
        if best.time > projectile_distance / ps.speed:
            return None
    return best
            
            