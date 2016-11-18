from math import pi
from math import hypot
from math import acos
from math import cos
from math import sin
from math import atan2
from math import sqrt
from copy import deepcopy
from Graph import Dijkstra
from model.CircularUnit import CircularUnit

EPSILON = 1E-4
MACRO_EPSILON = 2
MAX_OBSTACLES = 15
TARGET_EXTRA_SPACE = 50

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
   obstacles = [o for _, o in obstacles]
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
    if op.GetDistanceTo(p1) < o.radius - EPSILON or op.GetDistanceTo(p2) < o.radius - EPSILON:
        # print 1
        return True
    if p1.GetDistanceTo(p2) < EPSILON:
        return False
    line = Line(p1, p2)
    # print line
#     print op.GetDistanceToLine(line)
    if op.GetDistanceToLine(line) > o.radius - EPSILON:
        return False
    return (op - p1).ScalarMul(p2 - p1) * (op - p2).ScalarMul(p1 - p2) > 0
    

def SegmentClearFromObstacles(p1, p2, obstacles):
    for o in obstacles:
        if SegmentCrossesCircle(p1, p2, o):
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

def AddEdge(points, world, i1, i2, d, g, is_arc=False, circle=None):
    if Invalid(points[i1], world) or Invalid(points[i2], world):
        return
    g[i1].append((i2, d, is_arc, circle))
    g[i2].append((i1, d, is_arc, circle))
    if d < 0:
        print "ALARM"
        # import pdb; pdb.set_trace()
        
    if points[i1].GetDistanceTo(points[i2]) > d + EPSILON:
        print "WHHWAAAAAA %d %d %f" % (i1, i2, d)
        print points[i1]
        print points[i2]
        print is_arc
        print circle.x
        print circle.y
        print circle.radius
    
def GetArcLength(a1, a2, r):
    # print r
    # print a1
    # print a2
    return abs(a1 - a2) * r
    
def Downcast(ancestor, descendent):
    """
        automatic downcast conversion.....

        (NOTE - not type-safe -
        if ancestor isn't a super class of descendent, it may well break)

    """
    for name, value in vars(ancestor).iteritems():
        #print "setting descendent", name, ": ", value, "ancestor", name
        setattr(descendent, name, value)

    return descendent    

# target = (x, y)
# units = list(CircularUnit)
# TODO(vyakunin): decide on a better interface
# returns [(point, previous_point_no, shortest_distance)], points_per_unit: list(list(int)))
def FindOptimalPaths(me, units, world):
    me_point = deepcopy(me)
    me_point.radius = 0
    # import pdb; pdb.set_trace()
    circles = [CircularUnit(None, None, None, None, None, None, None, None)
               for _ in range(len(units))]
    for i, u in enumerate(units):
        circles[i] = Downcast(u, circles[i])
    units = circles
    # print units 
    all_units = deepcopy(units)
    for o in units:
        if abs(o.speed_x) + abs(o.speed_y) > EPSILON:
            new_o = deepcopy(o)
            new_o.x += o.speed_x
            new_o.y += o.speed_y
            all_units.append(new_o)
    for o in all_units:  
        o.radius = o.radius + me.radius + 5
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
                    # print 'distance: ' + str(p1) + str(p2) + str(distance)
                    AddEdge(points, world, i1, i2, distance, graph)

            intersections = IntersectCircles(u1, u2)
            if intersections is None:
               continue
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
            # if p[i-1][2] == 11 or p[i][2] == 11:
#                 import pdb; pdb.set_trace()
            # only add arc edges not passing intersection arcs. Keep those
            # inside intersections isolated from those outside, so they are unreachable.
            if p[i-1][1] == 0 and p[i][1] == 0:
                AddEdge(points, world, p[i-1][2], p[i][2], 
                        GetArcLength(p[i-1][0], p[i][0], all_units[unit_index].radius),
                        graph, is_arc=True, circle=u)
        # leave just list of point numbers
        points_per_unit[unit_index] = [k for _, __, k in p if k != -1]
    optimal_distances, previous_points = Dijkstra(graph)
    return (points, previous_points, optimal_distances, points_per_unit)
    
def CloseEnough(me, t, c):
    if c[1]:
        # import pdb; pdb.set_trace()
        return abs(me.get_distance_to_unit(c[2]) - c[2].radius) < MACRO_EPSILON
    # import pdb; pdb.set_trace()
    return abs(me.get_distance_to_unit(t) + me.get_distance_to_unit(c[0]) - 
        t.GetDistanceTo(c[0])) < MACRO_EPSILON
        
def BuildPathAngle(me, path):
    t = None
    c = None
    for i, b in enumerate(path[:-1]):
        if b[1] and b[2].radius < EPSILON:
            continue
        e = path[i+1]
        if b[0].GetDistanceTo(e[0]) < EPSILON:
            continue
        if CloseEnough(me, e[0], b):
            t = e[0]
            c = b
            
    if t is None:
        print 'no path from '
        print me
        return None
    if not c[1]:
        return me.get_angle_to_unit(t)
    c = c[2]
    me_p = Point.FromUnit(me)
    v_to_c = Point.FromUnit(c) - me_p
    v_to_t = t - me_p
    c1 = v_to_c.Rotate(pi/2)
    if c1.ScalarMul(v_to_t) > 0:
        # import pdb; pdb.set_trace()
        return c1.GetAngle() - me.angle
    c2 = v_to_c.Rotate(-pi/2)
    return c2.GetAngle() - me.angle
    
# returns [[point, is_arc, circle]]
def BuildPath(me, target, game, world):
    obstacles = BuildObstacles(me, world, game)
    target = CircularUnit(0, target.x, target.y, 0, 0, 0, 0, 0)
    obstacles = [o for o in obstacles if o.get_distance_to_unit(target) > 
                                         o.radius + me.radius + TARGET_EXTRA_SPACE]
    p, prev, d, points_per_unit = FindOptimalPaths(
        me, [target] + obstacles, world)
    # print obstacles
    # print 'points:'
    # print p
    # print 'prev:'
    # print prev
    # print 'optimal_distances:'
    # print d
    
    # import pdb; pdb.set_trace()
    t_id = -1
    min_d = 1e6
    for ps in points_per_unit[1]:
        if t_id == -1 or d[ps] < min_d:
            min_d = d[ps]
            t_id = ps
    if t_id == -1:
        import pdb; pdb.set_trace()
        print 'no way to ' + str(target)
        return None
    # import pdb; pdb.set_trace()
    path = [(p[t_id], prev[t_id])]
    
    while prev[t_id][0] != -1:
        t_id = prev[t_id][0]
        path.append((p[t_id], prev[t_id]))
    path.reverse()
    
    for i, p in enumerate(path[:-1]):
        path[i] = [path[i][0]] + list(path[i+1][1][1:])
    return path