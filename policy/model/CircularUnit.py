from model.Unit import Unit


class CircularUnit(Unit):
    def __init__(self, id, x, y, speed_x, speed_y, angle, faction, radius):
        Unit.__init__(self, id, x, y, speed_x, speed_y, angle, faction)

        self.radius = radius
    def __str__(self):
        return 'CircularUnit(%d, %d, %d, %.3f, %.3f, %.3f, %d, %.3f)' % ( 
                self.id, self.x, self.y, self.speed_x, self.speed_y, 
                self.angle, self.faction, self.radius)
    
    def __repr__(self):
        return self.__str__()
