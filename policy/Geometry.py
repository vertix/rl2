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

EPSILON = 1E-4
MACRO_EPSILON = 2
MAX_OBSTACLES = 15
TARGET_EXTRA_SPACE = 50
INFINITY = 1e6
TREE_DISCOUNT = 2
WIZARD_DPS = 12.0 / 60.0 * TREE_DISCOUNT # missile every 60 ticks
WIZARD_SPEED = 4.0  # per tick, going forward
TICKS_TO_ACCOUNT_FOR = 10
RADIUS_ALLOWANCE = 4

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
            
    def GetTargetId(self):
        if self.type == Edge.STRAIGHT_SEGMENT:
            return self.edge.target_id
        else:
            return -1
        
    def IsNonTrivial(self):
        if self.type == Edge.ARC and self.circle.radius < EPSILON:
            return False
        if self.begin.GetDistanceTo(self.end) < EPSILON:
            return False
        return True
        
    def GetDistanceTo(self, p):
        a = Point.FromUnit(p)
        if (self.type == Edge.CLEAR_SEGMENT) or (self.type == Edge.STRAIGHT_SEGMENT):
            return a.GetDistanceToSegment(self.begin, self.end)
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
    
    def GetNextAngleDistanceAndTarget(self, u):
        current = self.GetCurrentTransition(u)
        if current is None:
            return None
        return current.GetAngleAndDistanceFrom(u) + (current.GetTargetId(),)
        

class Line(object):
    def __init__(self, p1, p2):
        self.a = p1.y - p2.y
        self.b = p2.x - p1.x
        self.c = -self.a * p1.x - self.b * p1.y
        
    def __str__(self):
        return '(%.6f*x + %.6f*y + %.6f = 0)' % (self.a, self.b, self.c)
    
    def __repr__(self):
        return self.__str__()

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @classmethod
    def FromUnit(cls, u):
        return cls(u.x, u.y)

    def Rotate(self, a):
        return Point(self.x * cos(a) - self.y * sin(a), self.x * sin(a) + self.y * cos(a))
        
    def GetDistanceTo(self, p):
        return (self - p).Norm()
        
    def ScalarMul(self, p):
        return self.x * p.x + self.y * p.y
    
    def Norm(self):
        return hypot(self.x, self.y)
        
    def GetDistanceToLine(self, l):
        return abs(self.x * l.a + self.y * l.b + l.c) / hypot(l.a, l.b)
        
    def GetDistanceToSegment(self, p1, p2):
        ans = min(self.GetDistanceTo(p1), self.GetDistanceTo(p2))
        if ((self - p1).ScalarMul(p2 - p1) > 0) and ((self - p2).ScalarMul(p1 - p2) > 0):
            ans = min(ans, self.GetDistanceToLine(Line(p1, p2)))
            
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
    

def BuildObstacles(me, world, game):
   obstacles = [(me.get_distance_to_unit(unit), deepcopy(unit)) for unit in 
       world.wizards + world.minions + world.trees + world.buildings 
       if me.get_distance_to_unit(unit) < me.vision_range and me.id != unit.id]
   obstacles.sort()
   obstacles = [Obstacle(me, o) for _, o in obstacles]
   obstacles = obstacles[:MAX_OBSTACLES]
   # make all objects larger by error margin + my radius to be a point
   return obstacles


def BuildTangentPoint(u, v, alpha):
    v_rotated = v.Rotate(alpha)
    return (v_rotated + u, v_rotated.GetAngle())


# u1, u2: CircularUnit
# returns: list((point1, alpha1, point2, alpha2)) of tangents with pointX on
# uX having angle alphaX.(-pi <= alphaX < pi)
def BuildTangents(u1, u2):
    r1 = u1.radius
    r2 = u2.radius
    d = u1.get_distance_to_unit(u2)
    u1p = Point.FromUnit(u1)
    u2p = Point.FromUnit(u2)
    if r1 > r2:
        return [(p2, a2, p1, a1) for (p1, a1, p2, a2) in BuildTangents(u2, u1)]
    # now r1 < r2
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
    
def SegmentCrossesCircle(p1, p2, o):
    if (o.radius < EPSILON):
        return False
    op = Point.FromUnit(o)
    return op.GetDistanceToSegment(p1, p2) < o.radius - EPSILON


def SegmentClearFromObstacles(p1, p2, obstacles, ignore_id=-1):
    for i, o in enumerate(obstacles):
        if (i != ignore_id) and SegmentCrossesCircle(p1, p2, o):
            # import pdb; pdb.set_trace()
            return False
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
    
    assert abs(p1.GetDistanceTo(c1p) - c1.radius) < EPSILON
    assert abs(p2.GetDistanceTo(c1p) - c1.radius) < EPSILON
    assert abs(p1.GetDistanceTo(c2p) - c2.radius) < EPSILON
    assert abs(p2.GetDistanceTo(c2p) - c2.radius) < EPSILON
    
    return ([p1, p2], [((p1 - c1p).GetAngle(), (p2 - c1p).GetAngle()), 
                       ((p1 - c2p).GetAngle(), (p2 - c2p).GetAngle())])
                       
# a1, a2 - starting and ending angles for arc, i1 and i2 - corresponding point numbers,
# returns [(-pi <= alpha < pi, type, point_no)...],
# type is 1 for beginning of intersection and -1 for end of intersection. point_no = -1 
# means fake point.
def AddArc(a1, a2, i1, i2):
    if a1 > a2 + EPSILON:
        # break arc into two to wrap pi point, assign fake point ID.
        return [(a1, 1, i1), (pi, -1, -1), (-pi, 1, -1), (a2, -1, i2)]
    return [(a1, 1, i1), (a2, -1, i2)]
    
def Invalid(p, world):
    r = world.wizards[0].radius
    return p.x < r or p.x > world.width - r or p.y < r or p.y > world.height - r
    
class Edge(object):
    CLEAR_SEGMENT = 0
    ARC = 1
    STRAIGHT_SEGMENT = 2
    def __init__(self, v, w, type, circle, target_id=-1):
        self.v = v
        self.w = w
        self.type = type
        self.circle = circle
        self.target_id = target_id

def AddEdge(points, world, i1, i2, d, g, type=Edge.CLEAR_SEGMENT, circle=None, target_id=-1):
    if Invalid(points[i1], world) or Invalid(points[i2], world):
        return
    g[i1].append(Edge(i2, d, type, circle, target_id))
    g[i2].append(Edge(i1, d, type, circle, target_id))
    assert d >= 0

    # if points[i1].GetDistanceTo(points[i2]) > d + EPSILON:
    #     print "WHHWAAAAAA %d %d %f" % (i1, i2, d)
    #     print points[i1]
    #     print points[i2]
    #     print is_arc
    #     print circle.x
    #     print circle.y
    #     print circle.radius

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
        CircularUnit.__init__(self, u.id, u.x, u.y, u.speed_x, u.speed_y, u.angle, u.faction, u.radius)
        self.straight_penalty = INFINITY
        if me.faction != u.faction:
            self.straight_penalty = u.life / WIZARD_DPS * WIZARD_SPEED
            
# returns segment1, segment2, where each segment is [(alpha1, point1), (alpha2, point2)]
# for corresponding circle
def BuildStraightSegments(c1, c2):
    c2p = Point.FromUnit(c2)
    c1p = Point.FromUnit(c1)
    v12 = c2p - c1p
    if v12.Norm() < EPSILON:
        return ([(0, c1p), (0, c1p)], [(0, c2p), (0, c2p)])
    v12 *= 1 / v12.Norm()
    p11 = c1p + (v12 * (-c1.radius))
    p12 = c1p + (v12 * (c1.radius))

    p21 = c2p + (v12 * (-c2.radius))
    p22 = c2p + (v12 * (c2.radius))
    return ([((p11 - c1p).GetAngle(), p11),
             ((p12 - c1p).GetAngle(), p12)
            ],
            [((p21 - c2p).GetAngle(), p21),
             ((p22 - c2p).GetAngle(), p22)])

# target = (x, y)
# units = list(CircularUnit)
# TODO(vyakunin): decide on a better interface
# returns [(point, previous_point_no, shortest_distance)], points_per_unit: list(list(int)))
def FindOptimalPaths(me, units, world):
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
                if SegmentClearFromObstacles(p1, p2, all_units):
                    i1 = len(points)
                    i2 = i1 + 1
                    points.append(p1)
                    points.append(p2)
                    graph.append([])
                    graph.append([])    
                    # if i1 == 11 or i2 == 11:
                    #     import pdb; pdb.set_trace()
                    points_per_unit[i].append((a1, 0, i1))
                    points_per_unit[j].append((a2, 0, i2))
                    distance = p1.GetDistanceTo(p2)
                    AddEdge(points, world, i1, i2, distance, graph)

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
            
            # segment = [(alpha1, point1), (alpha2, point2)]
            if i > (len(all_units) + 4) / 5:
                continue
            if u2.straight_penalty > INFINITY / 2:
                continue
            if (i > 0) and (u2.straight_penalty > INFINITY / 2):
                continue
            segment1, segment2 = BuildStraightSegments(u1, u2)
            if SegmentClearFromObstacles(segment1[0][1], segment1[1][1], all_units, i):
                i11 = len(points)
                i12 = i11 + 1
                points.extend([p[1] for p in segment1])
                graph.append([])
                graph.append([])
                points_per_unit[i].extend([(segment1[0][0], 0, i11), (segment1[1][0], 0, i12)])
                AddEdge(points, world, i11, i12,
                        segment1[0][1].GetDistanceTo(segment1[1][1]) + u1.straight_penalty, 
                        graph, Edge.STRAIGHT_SEGMENT, None, u1.id)
            if SegmentClearFromObstacles(segment2[0][1], segment2[1][1], all_units, j):
                i11 = len(points)
                i12 = i11 + 1
                points.extend([p[1] for p in segment2])
                graph.append([])
                graph.append([])
                points_per_unit[j].extend([(segment2[0][0], 0, i11), (segment2[1][0], 0, i12)])
                AddEdge(points, world, i11, i12,
                        segment2[0][1].GetDistanceTo(segment2[1][1]) + u2.straight_penalty, 
                        graph, Edge.STRAIGHT_SEGMENT, None, u2.id)
            
            
    for unit_index, p in enumerate(points_per_unit):
        u = all_units[unit_index]
        p.sort()
        for i in range(len(p)):
            # if p[i-1][2] == 11 or p[i][2] == 11:
#                 import pdb; pdb.set_trace()
            # only add arc edges not passing intersection arcs. Keep those
            # inside intersections isolated from those outside, so they are unreachable.
            if p[i-1][1] == 0 and p[i][1] == 0:
                AddEdge(points, world, p[i-1][2], p[i][2], 
                        GetArcLength(p[i-1][0], p[i][0], all_units[unit_index].radius),
                        graph, type=Edge.ARC, circle=u)
        # leave just list of point numbers
        points_per_unit[unit_index] = [k for _, __, k in p if k != -1]
        
    optimal_distances, prev = Dijkstra(graph)
    return (points, prev, optimal_distances, points_per_unit)
    

# returns [[point, is_arc, circle]]
def BuildPath(me, target, game, world):
    obstacles = BuildObstacles(me, world, game)
    if isinstance(target, LivingUnit):
        t = deepcopy(target)
    else:
        t = LivingUnit(0, target.x, target.y, 0, 0, 0, 0, 0, 0, 0, [])
    obstacles = [o for o in obstacles if o.get_distance_to_unit(target) > 
                                         o.radius + me.radius + TARGET_EXTRA_SPACE]
    p, prev, d, points_per_unit = FindOptimalPaths(
        me, [Obstacle(me, t)] + obstacles, world)
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
    
def HasMeleeTarget(u, world, game):
    a = None
    r = None
    if isinstance(u, Wizard):
        a = game.staff_sector
        r = game.staff_range
    if isinstance(u, Minion) and (u.type == MinionType.ORC_WOODCUTTER):
        a = orc_woodcutter_attack_sector
        r = orc_woodcutter_attack_range
    if r is None:
        return False
    for t in world.wizards + world.minions + world.buildings + world.trees:
        if (t.faction != u.faction) and SectorCovers(u, a, r, t):
            return True
    return False

def RangeAllowance(me, target):
    return me.cast_range - (me.get_distance_to_unit(target) - target.radius)

def TargetInRangeWithAllowance(me, target, allowance):
    return RangeAllowance(me, target) > allowance

