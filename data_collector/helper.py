import cv2
import numpy as np
import os
import signal

CAM_INDEX = 1
on = True

ZERO = 0
ONE = 1
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

    if not os.path.exists("../data/one") or not os.path.isdir("../data/one"):
        os.mkdir("../data/one")

    if not os.path.exists("../data/two") or not os.path.isdir("../data/two"):
        os.mkdir("../data/two")

    if not os.path.exists("../data/three") or not os.path.isdir("../data/three"):
        os.mkdir("../data/three")
        
    if not os.path.exists("../data/four") or not os.path.isdir("../data/four"):
        os.mkdir("../data/four")
        
    if not os.path.exists("../data/five") or not os.path.isdir("../data/five"):
        os.mkdir("../data/five")


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
        if key == ord('e'):
            on = False
            break
        elif key == ord(str(ZERO)):
            print("Zero")
            path = "../data/zero"
        elif key == ord(str(ONE)):
            print("One")
            path = "../data/one"
        elif key == ord(str(TWO)):
            print("Two")
            path = "../data/two"
        elif key == ord(str(THREE)):
            print("Three")
            path = "../data/three"
        elif key == ord(str(FOUR)):
            print("Four")
            path = "../data/four"
        elif key == ord(str(FIVE)):
            print("Five")
            path = "../data/five"
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
