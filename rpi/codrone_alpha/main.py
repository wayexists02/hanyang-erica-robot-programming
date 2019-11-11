#!/usr/bin/env python3

import os
import sys
import time
import signal


on = True


def interrupt_handler(signo, frame):
    global on
    on = False
    exit(0)


def main(argv):
    in_fd = int(argv[1])
    out_fd = int(argv[2])

    print("IN FD", in_fd)
    print("OUT FD", out_fd)

    while on:
        msg = os.read(in_fd, 16)
        print("MSG: {}".format(msg))
        time.sleep(0.1)

    os.close(in_fd)
    os.close(out_fd)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt_handler)

    main(sys.argv)