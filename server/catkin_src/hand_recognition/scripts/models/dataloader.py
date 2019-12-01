import numpy as np
import pandas as pd
import cv2
import pathlib
import os
from env import *


TRAIN_PATH = NOTHING_TRAIN_PATH
VALID_PATH = NOTHING_VALID_PATH
CAT = NOTHING_CAT

# TRAIN_PATH = SIGN_TRAIN_PATH
# VALID_PATH = SIGN_VALID_PATH
# CAT = SIGN_CAT


class DataLoader():

    def __init__(self, batch_size, train=True, noise=True, flip=True):
        if train:
            self.data_path = TRAIN_PATH
        else:
            self.data_path = VALID_PATH

        self.noise = noise
        self.flip = flip

        self.data_list = []
        for i, cat_list in enumerate(self._read_data_list()):
            self.data_list.extend(list(map(lambda filename: (filename, i), cat_list)))

        self.n = len(self.data_list)

        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n / self.batch_size))

        self.min_lim = (0, 10, 10)
        self.max_lim = (45, 240, 255)

    def __len__(self):
        return self.num_batches

    def next_batch(self):

        n_classes = len(CAT)
        
        shuffled_data_list = self.data_list.copy()
        np.random.shuffle(shuffled_data_list)

        for b in range(self.num_batches):
            start = b*self.batch_size
            end = min((b+1)*self.batch_size, self.n)

            x_batch = np.zeros(((end - start), 1, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start),))

            for i in range(start, end):
                x = shuffled_data_list[i][0]
                y = shuffled_data_list[i][1]

                x_batch[i - start] = self._load_data(x)
                y_batch[i - start] = y

            yield x_batch, y_batch
        
    def _read_data_list(self):
        dat_list = []

        for c in CAT:
            cat_list = []
            for f in pathlib.Path(os.path.join(self.data_path, c).replace("\\", "/")).glob("*.jpg"):
                cat_list.append(str(f))

            dat_list.append(cat_list)
            print(c, len(cat_list))

        return dat_list

    def _load_data(self, path):
        img = cv2.imread(path)
        
        img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img = img.astype(np.int32)

        if self.noise is True and np.random.rand() < 0.5:
            img = img + np.random.randn(*img.shape) * 3
            img[img < 0] = 0
            img[img > 255] = 255

        img = img.astype(np.uint8)

        bitmask = cv2.inRange(img, np.array([0, 10, 60]), np.array([20, 180, 255]))

        img = img * bitmask.reshape(*bitmask.shape, 1)

        # if self.flip is True:
        #     if np.random.rand() < 0.5:
        #         img = img[::-1, :]
        #     if np.random.rand() < 0.5:
        #         img = img[:, ::-1]


        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(*img.shape, 1)

        img = img.astype(np.float32)
        # print(img.max(), img.min())
        img = (img - 128) / 256
        # print(img.shape)

        img = np.transpose(img, (2, 0, 1))
        return img
