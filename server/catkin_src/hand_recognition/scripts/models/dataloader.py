import numpy as np
import pandas as pd
import cv2
import pathlib
import os
from env import *


# TRAIN_PATH = NOTHING_TRAIN_PATH
# VALID_PATH = NOTHING_VALID_PATH
# CAT = NOTHING_CAT

TRAIN_PATH = SIGN_TRAIN_PATH
VALID_PATH = SIGN_VALID_PATH
CAT = SIGN_CAT


class DataLoader():
    """
    Hand gesture classifier 를 학습시키기 위한 데이터를 로드하는 객체.
    """

    def __init__(self, batch_size, train=True, noise=True, flip=True):
        if train:                   # trainset path
            self.data_path = TRAIN_PATH
        else:                       # validset path
            self.data_path = VALID_PATH

        self.noise = noise          # 이미지에 가우시안 노이즈 추가 여부
        self.flip = flip            # 이미지를 상하좌우 랜덤 플립 적용 여부

        # 데이터셋의 이미지 파일들의 파일명 리스트를 읽어옴. 이미지는 아직 로드 안함.
        self.data_list = []
        for i, cat_list in enumerate(self._read_data_list()):
            self.data_list.extend(list(map(lambda filename: (filename, i), cat_list)))

        # 이미지 개수
        self.n = len(self.data_list)

        # 배치 크기 및 배치의 개수 계산
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n / self.batch_size))

    def __len__(self):
        return self.num_batches

    def next_batch(self):
        """
        배치를 생산하는 generator 함수.

        Yields:
        -------
        x_batch : 이미지 배치 (batch_size, channel, height, width)
        y_batch : 라벨        (batch_size,)
        """

        # 카테고리 개수
        n_classes = len(CAT)
        
        # 데이터 리스트의 복사본을 만들고 셔플함
        shuffled_data_list = self.data_list.copy()
        np.random.shuffle(shuffled_data_list)

        # 배치 개수 만큼 반복
        for b in range(self.num_batches):
            start = b*self.batch_size                       # 배치 시작 인덱스
            end = min((b+1)*self.batch_size, self.n)        # 배치 끝 인덱스

            # 배치가 들어갈 placeholder
            x_batch = np.zeros(((end - start), 3, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start),))

            # 이미지를 한 장씩 로드하고 배치 placeholder에 저장
            for i in range(start, end):
                x = shuffled_data_list[i][0]
                y = shuffled_data_list[i][1]

                x_batch[i - start] = self._load_data(x)
                y_batch[i - start] = y

            # 배치 생성
            yield x_batch, y_batch
        
    def _read_data_list(self):
        """
        데이터셋 안에 있는 데이터들의 파일명 리스트를 반환

        Returns:
        --------
        dat_list : 데이터 파일명 리스트
        """

        dat_list = []

        for c in CAT:
            cat_list = []
            for f in pathlib.Path(os.path.join(self.data_path, c).replace("\\", "/")).glob("*.jpg"):
                cat_list.append(str(f))

            dat_list.append(cat_list)
            print(c, len(cat_list))

        return dat_list

    def _load_data(self, path):
        """
        이미지 한 장을 로드함

        Arguments:
        ----------
        path : 이미지 한 장의 path

        Returns:
        --------
        img : 이미지 matrix
        """
        img = cv2.imread(path)
        
        # 크기 맞춰주고 RGB로 변경
        img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 랜덤 플립 (상하좌우)
        if self.flip is True:
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]

        img = img.astype(np.int32)

        # 랜덤 가우시안 노이즈
        if self.noise is True and np.random.rand() < 0.5:
            img = img + np.random.randn(*img.shape) * 2
            img[img < 0] = 0
            img[img > 255] = 255

        # standardization
        img = img.astype(np.float32)
        img = (img - 128) / 256

        # pytorch 입력에 맞게 transpose
        img = np.transpose(img, (2, 0, 1))
        return img
