from Geometry import BuildTangents
from Geometry import SegmentCrossesCircle
from Geometry import Point
from model.CircularUnit import CircularUnit
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



