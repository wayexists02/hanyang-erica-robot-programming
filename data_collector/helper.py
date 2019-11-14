import cv2
import numpy as np
import os
import signal

CAM_INDEX = 0
on = True

NOTHING = "n"
LANDING = 0
TAKE_OFF = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5

index = 0


def handle_interrupt(signo, frame):
    global on
    on = False


def mkdir():
    if not os.path.exists("../data") or not os.path.isdir("../data"):
        os.mkdir("../data")

    if not os.path.exists("../data/zero") or not os.path.isdir("../data/zero"):
        os.mkdir("../data/zero")

    if not os.path.exists("../data/landing") or not os.path.isdir("../data/landing"):
        os.mkdir("../data/landing")

    if not os.path.exists("../data/takeoff") or not os.path.isdir("../data/takeoff"):
        os.mkdir("../data/takeoff")

    if not os.path.exists("../data/2") or not os.path.isdir("../data/2"):
        os.mkdir("../data/2")

    if not os.path.exists("../data/3") or not os.path.isdir("../data/3"):
        os.mkdir("../data/3")

    if not os.path.exists("../data/4") or not os.path.isdir("../data/4"):
        os.mkdir("../data/4")

    if not os.path.exists("../data/5") or not os.path.isdir("../data/5"):
        os.mkdir("../data/5")


def read_info():
    global index
    index = 0

    if not os.path.exists("./index.txt") or not os.path.isfile("./index.txt"):
        with open("./index.txt", "w") as f:
            f.write(str(index))

    else:
        with open("./index.txt", "r") as f:
            index = int(f.readline())


def main():
    global on
    global index

    mkdir()
    read_info()

    cap = cv2.VideoCapture(CAM_INDEX)

    while on:
        check, frame = cap.read()
        if check is False:
            continue

        frame = cv2.resize(frame, dsize=(256, 256))
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(10)
        if key == ord('e') or key == ord('E'):
            on = False
            break
        elif key == ord('n') or key == ord('N'):
            print("Nothing (N)")
            path = "../data/nothing"
        elif key == ord(str(LANDING)):
            print("Landing (0)")
            path = "../data/landing"
        elif key == ord(str(TAKE_OFF)):
            print("Takeoff (1)")
            path = "../data/takeoff"
        elif key == ord(str(TWO)):
            print("2")
            path = "../data/2"
        elif key == ord(str(THREE)):
            print("3")
            path = "../data/3"
        elif key == ord(str(FOUR)):
            print("4")
            path = "../data/4"
        elif key == ord(str(FIVE)):
            print("5")
            path = "../data/5"
        else:
            continue

        filename = "%08d.jpg" %index
        path = os.path.join(path, filename).replace("\\", "/")

        cv2.imwrite(path, frame)
        print("Image %s was saved in %s" %(filename, path))

        index += 1

    cv2.destroyAllWindows()
    with open("./index.txt", "w") as f:
        f.write(str(index))

    print("Terminated.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    main()
