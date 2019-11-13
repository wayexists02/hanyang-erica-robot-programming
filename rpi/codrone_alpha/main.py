#!/usr/bin/env python3

import os
import sys
import time
import signal
import json

from codrone_alpha import CoDroneAlpha


on = True


def interrupt_handler(signo, frame):
    global on
    on = False
    exit(0)


def main(argv):
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

        # json 문자열로 변환
        json_data = json.dumps(data)

        # 데이터의 길이를 8바이트 문자열로 생성
        buf = "%8d" % len(json_data)

        # 데이터의 길이를 먼저 전송
        os.write(out_fd, buf.encode("utf-8"))

        # 데이터 전송
        os.write(out_fd, json_data.encode("utf-8"))

        # codrone.update_data() 에서 이미 sleep이 길다.
        # time.sleep(0.1)

    os.close(in_fd)
    os.close(out_fd)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt_handler)

    main(sys.argv)