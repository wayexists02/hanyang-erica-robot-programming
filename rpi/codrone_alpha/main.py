#!/usr/bin/env python3

import os
import sys
import time
import signal

from codrone_alpha import CoDroneAlpha


on = True
codrone = None


def interrupt_handler(signo, frame):
    codrone.drone.sendStop()
    exit(0)


def main(argv):
    global codrone

    in_fd = int(argv[1])
    out_fd = int(argv[2])

    # TEST
    # print("IN FD", in_fd)
    # print("OUT FD", out_fd)

    # 드론 객체 생성
    codrone = CoDroneAlpha()
    codrone.init()

    while on:
        # TEST
        # msg = os.read(in_fd, 16)
        # print("MSG: {}".format(msg))

        # 코드론의 정보를 받아옴
        codrone.update_data()
        data = codrone.get_data()

        if data is None:
            continue

        # 데이터의 길이를 8바이트 문자열로 생성
        buf = "%8d" % len(data)

        # 데이터의 길이를 먼저 전송
        os.write(out_fd, buf.encode("utf-8"))

        # 데이터 전송
        os.write(out_fd, data.encode("utf-8"))
        
        # 명령 길이 정보 받기
        len_of_cmd = int(os.read(in_fd, 8).decode("utf-8"))

        # 명령 받기
        cmd = os.read(in_fd, len_of_cmd).decode("utf-8")

        codrone.send_command(cmd)

    os.close(in_fd)
    os.close(out_fd)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt_handler)

    main(sys.argv)