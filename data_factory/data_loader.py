import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + "/train.csv")
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + "/test.csv")

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + "/test_label.csv").values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            )


class HMCSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()

        train_df = pd.read_csv(data_path + "/train.csv")
        # train_df = train_df[:1000]

        data_columns = train_df.columns.drop(["Timestamp", "anomaly"])
        train_df = train_df[data_columns].astype(float)
        scaler = self.scaler.fit(train_df)
        train = scaler.transform(train_df)
        train_df = pd.DataFrame(train, columns=train_df.columns, index=list(train_df.index.values))
        train_df = train_df.ewm(alpha=0.9).mean()

        test_df = pd.read_csv(data_path + "/test.csv")
        # test_df = test_df[:1000]

        test_df = test_df[data_columns].astype(float)
        test = scaler.transform(test_df)
        test_df = pd.DataFrame(test, columns=test_df.columns, index=list(test_df.index.values))
        test_df = test_df.ewm(alpha=0.9).mean()

        self.train_df = train_df
        self.test_df = test_df
        self.train = np.array(train_df.values)
        self.test = np.array(test_df.values)
        self.valid = self.test
        self.test_labels = np.zeros(self.test.shape[0])

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.valid.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            )


def check_dataframe(df, mode="train"):
    x = np.array(df, dtype=np.float32)
    over_1 = np.any(x > 1.0)
    below_0 = np.any(x < 0.0)
    is_nan = np.any(np.isnan(x))
    print(f"Any {mode} data over 1.0: {over_1}, below 0: {below_0}, none: {is_nan}")


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode="train", dataset="KDD"):
    if dataset == "SMD":
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == "MSL":
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == "SMAP":
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == "PSM":
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset == "HMC":
        dataset = HMCSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == "train":
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader
