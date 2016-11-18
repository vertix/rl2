from Geometry import BuildTangents
from Geometry import SegmentCrossesCircle
from Geometry import Point
from Geometry import FindOptimalPaths
from Geometry import BuildNextAngle
from model.CircularUnit import CircularUnit
from model.World import World
from model.Wizard import Wizard
from model.Game import Game
u1 = CircularUnit(0, 1, 1, 0, 0, 0, 0, 1)
u2 = CircularUnit(0, 2, 4, 0, 0, 0, 0, 1)
print BuildTangents(u1, u2)
print '\n\n\n\n\n'

assert not SegmentCrossesCircle(Point(1, 2), Point(1, 6), u2)
assert not SegmentCrossesCircle(Point(0, 2), Point(0, 6), u2)
assert SegmentCrossesCircle(Point(2, 2), Point(2, 6), u2)
assert SegmentCrossesCircle(Point(2, 2), Point(2, 4), u2)
assert SegmentCrossesCircle(Point(2, 4), Point(2, 6), u2)
assert SegmentCrossesCircle(Point(2, 4), Point(2, 5), u2)
assert SegmentCrossesCircle(Point(2, 1), Point(2, 6), u2)
assert SegmentCrossesCircle(Point(0, 2), Point(5, 5), u2)
assert not SegmentCrossesCircle(Point(3, 1), Point(3, 6), u2)

me = Wizard(1, 100, 3700, 0.000, 0.000, -1.571, 0, 35, 0, 0, 0, 0,
                 True, 0, 0, 600, 0, 0, 0, 0, 0,
                 0, 0, 0)
target = CircularUnit(0, 1333, 2666, 0.000, 0.000, 0.000, 0, 0.000)
obstacles = [CircularUnit(2, 200, 3700, 0.000, 0.000, -1.571, 0, 35), CircularUnit(3, 200, 3800, 0.000, 0.000, -0.785, 0, 35), CircularUnit(4, 300, 3800, 0.000, 0.000, 0.000, 0, 35), CircularUnit(5, 300, 3900, 0.000, 0.000, 0.000, 0, 35), CircularUnit(11, 400, 3600, 0.000, 0.000, 0.000, 0, 100.000)]
world = World(0, 0, 4000, 4000, None, [me], [], None, None,
                 [], obstacles)
game = Game(0, 0, 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 35, 0,
                 600, 4, 
                 3, 3,
                 0, 0, 0, 0,
                 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0, 0,
                 0, 0,
                 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0,
                 0, 0)
assert abs(BuildNextAngle(me, target, game, world) - 0.000203673205103) < 1e-4



me = Wizard(1, 164, 3477, 0.000, 0.000, 0.075, 0, 35, 0, 0, 0, 0,
                 True, 0, 0, 600, 0, 0, 0, 0, 0,
                 0, 0, 0)


obstacles = [CircularUnit(2, 236, 3471, -1.754, 3.070, -1.695, 0, 35), CircularUnit(11, 400, 3600, 0.000, 0.000, 0.000, 0, 100.000), CircularUnit(3, 493, 3784, -3.040, 1.806, 0.126, 0, 35), CircularUnit(593, 426, 3068, 0.000, 0.000, 0.000, 3, 35), CircularUnit(4, 588, 3785, 0.000, 0.000, 0.069, 0, 35), CircularUnit(5, 582, 3856, 0.000, 0.000, -0.256, 0, 35)]
world = World(0, 0, 4000, 4000, None, [me], [], None, None,
                 [], obstacles)
assert abs(BuildNextAngle(me, target, game, world) - (-1.17720544189)) < 1e-4

me = Wizard(1, 1911, 2128, -2.701, -2.281, 0.163, 0, 35, 0, 0, 0, 0,
                 True, 0, 0, 600, 0, 0, 0, 0, 0,
                 0, 0, 0)

obstacles = [CircularUnit(3, 1963, 2179, 0.000, 0.000, -1.055, 0, 35.000), CircularUnit(692, 1859, 2035, -2.711, 1.286, -0.443, 0, 25.000), CircularUnit(693, 1989, 2235, -2.762, 1.171, -0.407, 0, 25.000), CircularUnit(691, 1750, 2237, -1.154, 2.769, -1.175, 0, 25.000), CircularUnit(690, 1679, 2241, -1.204, 2.748, -1.158, 0, 25.000), CircularUnit(13, 1929, 2400, 0.000, 0.000, 0.000, 0, 50.000), CircularUnit(702, 2114, 1914, 0.000, 0.000, 2.699, 1, 25.000), CircularUnit(415, 2016, 2507, 0.000, 0.000, 0.000, 3, 47.426), CircularUnit(704, 2251, 1903, 2.212, -2.027, 2.400, 1, 25.000), CircularUnit(539, 1504, 2045, 0.000, 0.000, 0.000, 3, 37.235), CircularUnit(506, 1823, 2578, 0.000, 0.000, 0.000, 3, 31.839), CircularUnit(705, 2245, 1808, 2.789, -1.105, 2.764, 1, 25.000), CircularUnit(703, 2201, 1751, 2.257, -1.976, 2.343, 1, 25.000), CircularUnit(465, 1966, 2610, 0.000, 0.000, 0.000, 3, 37.696), CircularUnit(529, 1468, 1924, 0.000, 0.000, 0.000, 3, 40.072)]
world = World(0, 0, 4000, 4000, None, [me], [], None, None,
                 [], obstacles)
                
print BuildNextAngle(me, target, game, world)
# 2.10893308804