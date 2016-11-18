from Actions import GetNextWaypoint
from Geometry import Point                                                  
from model.CircularUnit import CircularUnit                                 

print GetNextWaypoint([Point(200.000, 3800.000), Point(200.000, 3333.333), Point(200.000, 2666.667), Point(200.000, 2000.000), Point(200.000, 1333.333), Point(666.667, 200.000), Point(1333.333, 200.000), Point(2000.000, 200.000), Point(2666.667, 200.000), Point(3800.000, 200.000)], CircularUnit(1, 204, 3341, 0.000, 0.000, -3.000, 0, 35.000))