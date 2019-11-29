BATCH_SIZE = 64
ETA = 1e-3
EPOCHS = 30

NOTHING_TRAIN_PATH = "./data/nothing_or_sign/train"
NOTHING_VALID_PATH = "./data/nothing_or_sign/valid"
SIGN_TRAIN_PATH = "./data/sign_detection/train"
SIGN_VALID_PATH = "./data/sign_detection/valid"

WIDTH = 128
HEIGHT = 128

NOTHING_CAT = [
    "nothing",
    "sign"
]

SIGN_CAT = [
    "landing",
    "takeoff",
    "2",
    "3",
    "4",
    "5"
]

SIGN_CLF_CKPT_PATH = "/home/jylee/catkin_ws/src/hand_recognition/scripts/ckpts/sign_clf.pth"
NOTHING_CLF_CKPT_PATH = "/home/jylee/catkin_ws/src/hand_recognition/scripts/ckpts/nothing_clf.pth"