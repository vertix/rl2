import collections
import sys

import numpy as np
import zmq

GAMMA = 0.999
LEARNING_RATE = 0.01


def Q(coeff, state, action):
    result = coeff[action].dot(state)
    if np.isnan(result):
        print coeff[action]
        print action
        print state
    return result

def V(coeff, state, max_actions):
    return max(Q(coeff, state, a) for a in range(max_actions))

def main(argv):
    exp_socket_addr = argv[0]
    max_actions = int(argv[1])
    strat_socket_addr = argv[2]

    print 'Totally have %d actions' % max_actions

    context = zmq.Context()
    sock_exp = context.socket(zmq.REP)
    sock_exp.bind(exp_socket_addr)

    sock_strat = context.socket(zmq.PUB)
    sock_strat.bind(strat_socket_addr)

    coeff, old_coeff = None, None
    step = 0

    while True:
        msg = sock_exp.recv_pyobj()
        sock_exp.send('Ok')

        if coeff is None:
            coeff = [np.zeros(msg['s'].shape)
                     for _ in range(max_actions)]
            old_coeff = [np.zeros(msg['s'].shape)
                         for _ in range(max_actions)]

        assert not np.any(np.isnan(msg['s']))

        td_error = (msg['r'] + GAMMA * V(old_coeff, msg['s1'], max_actions) -
                    Q(coeff, msg['s'], msg['a']))
        td_error = np.clip(td_error, -10., 10.)
        # print td_error
        assert not np.isnan(td_error), msg
        coeff[msg['a']] += LEARNING_RATE * td_error * msg['s']

        step += 1
        if step % 1000 == 0:
            print '%dk steps' % (step / 1000)
            old_coeff = [c.copy() for c in coeff]
            print old_coeff
            if step > 3900:
                sock_strat.send_pyobj(old_coeff)


if __name__ == "__main__":
    main(sys.argv[1:])
