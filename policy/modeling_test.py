""" Tests for modelling logic
"""
import unittest

from model.Minion import Minion
from model.Wizard import Wizard
import State as state
import Geometry as geom


class MockUnit(object):
    def __init__(self, pos):
        self.x, self.y = pos[0], pos[1]

    def get_distance_to(self, x, y):
        return geom.Point(x, y).GetDistanceTo(geom.Point(self.x, self.y))

class MockState(object):
    def __init__(self, type, friend, pos):
        self.enemy = 0. if friend else 1.
        self.unit = MockUnit(pos)
        self.type = type
        self.rel_position = pos
        if type == state.LUType.MINION:
            self.hp = 30
            self.damage_per_tick = 0.5
            self.max_speed = 10.
            self.attack_range = 5.
            self.hp_regen = 0.
        elif type == state.LUType.WIZARD:
            self.hp = 100
            self.damage_per_tick = 1.
            self.max_speed = 10.
            self.attack_range = 100.
            self.hp_regen = 0.1
        else:
            self.hp = 300
            self.damage_per_tick = 0.5
            self.max_speed = 0.
            self.attack_range = 150.
            self.hp_regen = 0.

        self.max_hp = self.hp


class ModelingUnittest(unittest.TestCase):
    def setUp(self):
        pass

    def Minion(self, pos, friend=True):
        return MockState(state.LUType.MINION, friend, pos)

    def Building(self, pos, friend=True):
        return MockState(state.LUType.BUILDING, friend, pos)

    def Wizard(self, pos, friend=True):
        return MockState(state.LUType.WIZARD, friend, pos)

    def testMinionAgainstTower(self):
        minion = self.Minion((-10, -10))
        hostile_tower = self.Building((10, 10), False)
        friends = [minion]
        enemies = [hostile_tower]

        result = state.ModelEngagement(friends, enemies)

        self.assertTrue(minion in result)
        self.assertTrue(hostile_tower in result)

        print result[minion]

        self.assertLessEqual(result[minion][-1].hp, 0)
        self.assertGreater(
            result[minion][-1].damage_caused[state.LUType.BUILDING], 0)

        self.assertLessEqual(
            result[hostile_tower][-1].deaths_caused[state.LUType.MINION], 1.)
        self.assertLessEqual(
            result[hostile_tower][-1].damage_caused[state.LUType.MINION],
            minion.hp)

    def testWizardAgainstTowerFarAway(self):
        wizard = self.Wizard((0, 0))
        hostile_tower = self.Building((150, 150), False)

        result = state.ModelEngagement([wizard], [hostile_tower])

        # Wizard is friendly, not attacking, nothing happens
        self.assertEqual(0, len(result))

    def testHostileWizardAgainstTowerAndMinion(self):
        print '---------------------------'
        tower = self.Building((-150, -150))
        minion = self.Minion((-150, -200))
        hostile_wizard = self.Wizard((100, 100), False)

        result = state.ModelEngagement([minion, tower], [hostile_wizard])
        for k, v in result.iteritems():
            print '<<<<<<<<<   %s   >>>>>>>>>' % (state.UnitModel(k))
            for item in v:
                print '  ' + str(item)
            print ""
        print '---------------------------'




if __name__ == '__main__':
    unittest.main()
