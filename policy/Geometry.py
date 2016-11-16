from math import pi
from math import hypot
from math import acos
from math import cos
from math import sin
from math import atan2
from math import sqrt

EPSILON = 1E-4

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
        return self + other * -1

    def __str__(self):
        return '(x:%.6f, y:%.6f)' % (self.x, self.y)
    
    def __repr__(self):
        return self.__str__()
    

def BuildObstacles(me, world, game):
   obstacles = [unit.copy for unit in 
       world.wizards + world.minions + world.trees + world.buildings 
       if me.get_distance_to_unit(unit) < me.vision_range]
   # make all objects larger by error margin + my radius to be a point
   for o in obstacles:
       o.radius = o.radius + me.radius + hypot(o.speed_x, o.speed_y)
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
    if r1 > r2:
        return [(p2, a2, p1, a1) for (p1, a1, p2, a2) in BuildTangents(u2, u1)]
    # now r1 < r2
    if abs(r2) < EPSILON:
        # both circles are points
        return [(u1, 0, u2, 0)]
    if r2 > d + r1 - EPSILON:
        # they are inside each other
        return []
    result = []
    # vector from center1 to point on circle1 closest to circle2
    v1 = Point((u2.x - u1.x) * r1 / d, (u2.y - u1.y) * r1 / d)
    # same but for circle2
    v2 = Point((u1.x - u2.x) * r2 / d, (u1.y - u2.y) * r2 / d)    
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
    op = Point.FromUnit(o)
    if op.GetDistanceTo(p1) < o.radius - EPSILON or op.GetDistanceTo(p2) < o.radius:
        print 1
        return True
    if p1.GetDistanceTo(p2) < EPSILON:
        return False
    line = Line(p1, p2)
    print line
    print op.GetDistanceToLine(line)
    if op.GetDistanceToLine(line) > o.radius - EPSILON:
        return False
    return (op - p1).ScalarMul(p2 - p1) * (op - p2).ScalarMul(p1 - p2) > 0
    

def SegmentClearFromObstacles(p1, p2, obstacles):
    for o in obstacles:
        if SegmentCrossesCircle(p1, p2, o):
            return False
    return True
    
# returns ([p1, p2], [(a1, a2), (b1, b2)]) or None, where pX – points, aX – corresponding
# angles on c1, bX – corresponding angles on c2 intersection goes from a1 to a2 in positive
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
                       
# a1, a2 – starting and ending angles for arc, i1 and i2 – corresponding point numbers,
# returns [(-pi <= alpha < pi, type, point_no)...],
# type is 1 for beginning of intersection and -1 for end of intersection. point_no = -1 
# means fake point.
def AddArc(a1, a2, i1, i2):
    if a1 > a2 + EPSILON:
        # break arc into two to wrap pi point, assign fake point ID.
        return [(a1, 1, i1), (pi, -1, -1), (-pi, 1, -1), (a2, -1, i2)]
    return [(a1, 1, i1), (a2, -1, i2)]
    
def AddEdge(i1, i2, d, g):
    g[i1].append(i2, d)
    g[i2].append(i1, d)
    
# target = (x, y)
# units = list(CircularUnit)
# TODO(vyakunin): decide on a better interface
# returns [(point, previous_point_no, shortest_distance)], points_per_unit: list(list(int)))
def FindOptimalPaths(me, units):
   target = CircularUnit(id=0, x=target[0], y=target[1],
                         speed_x=0, speed_y=0, angle=0, faction=0, radius=0)
   me_point = Point.FromUnit(me)
   me_point.radius = 0
   
   all_units = [me_point] + units
   # points[i]: Point
   points = []
   # graph[point_no_i]: [(point_no_j, direct_distance_i_to_j), ...]}
   graph = [[]] * len(all_units)
   
   # points_per_unit[unit_no_i]: [(-pi <= alpha < pi, type, point_no)...],
   # type is 1 for beginning of intersection, 0 for regular point and -1 for end of 
   # intersection.
   points_per_unit = [[]] * len(all_units)
   for i, u1 in enumerate(all_units):
       for j, u2 in enumerate(all_units[i:]):
           j = i+j
           tangents = BuildTangents(u1, u2)
           for p1, a1, p2, a2 in tangents:
               if SegmentClearFromObstacles(p1, p2, obstacles):
                   i1 = len(points)
                   i2 = i1 + 1
                   points.append(p1)
                   points.append(p2)
                   points_per_unit[i].append((a1, 0, i1))
                   points_per_unit[j].append((a2, 0, i2))
                   distance = GetDistance(p1, p2)
                   AddEdge(i1, i2, distance, graph)
            
           intersections = IntersectCircles(u1, u2)
           if intersections is None:
               continue
           p, a = intersections
           i1 = len(points)
           i2 = i1 + 1
           points.extend(p)
           points_per_unit[i].extend(AddArc(a[0][0], a[0][1], i1, i2))
           points_per_unit[j].extend(AddArc(a[1][1], a[1][0], i2, i1))
    for unit_index, p in enumerate(points_per_unit):
        p.sort()
        for i in range(len(p)):
            # only add arc edges not passing intersection arcs. Keep those
            # inside intersections isolated from those outside, so they are unreachable.
            if p[i-1][1] == 0 and p[i][1] == 0:
                AddEdge(p[i-1][2], p[i][2], 
                        GetArcLength(p[i-1][0], p[i][0], 
                                     all_units[unit_index].radius),
                        graph)
        # leave just list of point numbers
        points_per_unit[unit_index] = [k for _, __, k in p if k != -1]
    optimal_distances, previous_points = Dijkstra(graph)
    return (zip(points, previous_points, optimal_distances), points_per_unit)