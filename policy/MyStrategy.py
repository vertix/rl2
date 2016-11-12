from model.ActionType import ActionType
from model.Game import Game
from model.Move import Move
from model.Wizard import Wizard
from model.World import World

import pdb


class State(object):
    def __init__(self, me, world, game):
        self.me = me
        self.world = world
        self.game = game


class Action(object):
    # NOOP, FLEE, PROCEED, ATTACK@X
    def __init__(self, move):
        self.move = move


class Policy(object):
    def Act(self, state):
        move = Move()
        
        move.speed = state.game.wizard_forward_speed
        move.strafe_speed = state.game.wizard_strafe_speed
        move.turn = state.game.wizard_max_turn_angle
        move.action = ActionType.MAGIC_MISSILE

        return Action(move)


class MyStrategy:
    def __init__(self):
        self.policy = Policy()
    
    def EncodeState(self, me, world, game):
        return State(me, world, game)

    def ImplementAction(self, action, move):
        for attr in ['strafe_speed'
                    ,'turn'
                    ,'action'
                    ,'cast_angle'
                    ,'min_cast_distance'
                    ,'max_cast_distance'
                    ,'status_target_id'
                    ,'skill_to_learn'
                    ,'messages']:
            setattr(move, attr, getattr(action.move, attr))
    
    def move(self, me, world, game, move):
        """
        @type me: Wizard
        @type world: World
        @type game: Game
        @type move: Move
        """
        state = self.EncodeState(me, world, game)
        action = self.policy.Act(state)
        self.ImplementAction(action, move)
        
