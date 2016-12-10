import argparse
import os
import sys
import time

from MyStrategy import MyStrategy
from RemoteProcessClient import RemoteProcessClient
from model.Move import Move

parser = argparse.ArgumentParser()
parser.add_argument('host', default="127.0.0.1", nargs='?')
parser.add_argument('port', default=31001, type=int, nargs='?')
parser.add_argument('token', default="0000000000000000", nargs='?')
parser.add_argument('--exp_socket', default=0, type=int)
parser.add_argument('--vars_socket', default=0, type=int)
parser.add_argument('--policy', default='default', choices=['default', 'nn', 'q'])
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--random_lane', action='store_true')
parser.add_argument('--n_step', action='store_true')
parser.add_argument('--dropout', default=1.0, type=float)

ARGS = parser.parse_args()


class Runner:
    def __init__(self):
        self.remote_process_client = RemoteProcessClient(ARGS.host, ARGS.port)
        self.token = ARGS.token

    def run(self):
        try:
            self.remote_process_client.write_token_message(self.token)
            self.remote_process_client.write_protocol_version_message()
            team_size = self.remote_process_client.read_team_size_message()
            game = self.remote_process_client.read_game_context_message()

            strategies = []

            for _ in xrange(team_size):
                strategies.append(MyStrategy(ARGS))

            while True:
                player_context = self.remote_process_client.read_player_context_message()
                if player_context is None:
                    break

                player_wizards = player_context.wizards
                if player_wizards is None or player_wizards.__len__() != team_size:
                    break

                moves = []

                for wizard_index in xrange(team_size):
                    player_wizard = player_wizards[wizard_index]

                    move = Move()
                    moves.append(move)
                    strategies[wizard_index].move(player_wizard, player_context.world, game, move)

                self.remote_process_client.write_moves_message(moves)
        finally:
            # END GAME
            for s in strategies:
                if hasattr(s, 'stop'):
                    # res_path = '../local-runner/result.txt'
                    res_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '../local-runner/result.txt')
                    print res_path
                    time.sleep(2.)
                    delta = time.time() - os.path.getmtime(res_path)
                    print delta
                    last_reward = None
                    if delta < 5.:
                        with open(res_path) as f:
                            lines = f.readlines()
                            result = lines[2].strip().split()
                            last_reward = int(result[1])
                    s.stop(last_reward)

            self.remote_process_client.close()


Runner().run()
