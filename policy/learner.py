import sys

import zmq


def main(argv):
    context = zmq.Context()
    sock_exp = context.socket(zmq.REP)
    sock_exp.bind(argv[0])

    while True:
        message = sock_exp.recv_pyobj()
        sock_exp.send('Ok')
        print 'Received message'


if __name__ == "__main__":
    main(sys.argv[1:])
