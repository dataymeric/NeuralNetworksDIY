import os
import sys

sys.path.append("../")
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from src.linear import Linear
from src.loss import *
from src.activation import (
    TanH,
    Sigmoid,
    StableSigmoid,
    Softmax,
    LogSoftmax,
    ReLU,
    TanH,
    Softplus,
)
from src.encapsulation import Sequential, Optim

np.random.seed(42)

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Data Loading                                                           │
    └────────────────────────────────────────────────────────────────────────┘
 """
from utils.mltools import *


def normalize_batch_image(X):
    mn = np.min(X)
    mx = np.max(X)
    X_norm = (X - mn) * (1.0 / (mx - mn))
    return X_norm


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(X) for X in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return normalize_batch_image(tmp[:, 1:]), tmp[:, 0].astype(int)


X_train, y_train = load_usps("../data/usps/USPS_train.txt")
X_test, y_test = load_usps("../data/usps/USPS_test.txt")

y_train_oh = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_oh = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
nb_class = y_train_oh.shape[1]
batch_size = 64

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Training small network                                                 │
    └────────────────────────────────────────────────────────────────────────┘
 """


def train_small_net():
    encoder = [
        Linear(256, 64),
        TanH(),
    ]
    decoder = [Linear(64, 256), Sigmoid()]
    net_simple = Sequential(*(encoder + decoder))
    optimizer = Optim(net_simple, BCELoss(), eps=1e-3)
    optimizer.SGD_eval(
        X_train,
        X_train,
        batch_size,
        30,
        test_size=0.33,
        return_dataframe=True,
        patience=None,
    )
    with open("../models/usps_30_epoch_simple_net.pkl", "wb") as f:
        pickle.dump(optimizer, f)


""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Training medium network                                                │
    └────────────────────────────────────────────────────────────────────────┘
"""


def train_medium_net():
    encoder = [
        Linear(256, 128),
        TanH(),
        Linear(128, 64),
        TanH(),
    ]
    decoder = [Linear(64, 128), TanH(), Linear(128, 256), Sigmoid()]
    net_cplx = Sequential(*(encoder + decoder))
    optimizer = Optim(net_cplx, BCELoss(), eps=1e-3)
    optimizer.SGD_eval(
        X_train,
        X_train,
        batch_size,
        30,
        test_size=0.33,
        return_dataframe=True,
        patience=None,
    )
    with open("../models/usps_30_epoch_medium_net.pkl", "wb") as f:
        pickle.dump(optimizer, f)


""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Small network with different activation function                       │
    └────────────────────────────────────────────────────────────────────────┘
 """


def per_acc_fct():
    d = {}
    for act_fct in [
        TanH,
        Sigmoid,
        StableSigmoid,
        Softmax,
        LogSoftmax,
        ReLU,
        TanH,
        Softplus,
    ]:
        try:
            encoder = [
                Linear(256, 64),
                act_fct(),
            ]
            decoder = [Linear(64, 256), Sigmoid()]
            net = Sequential(*(encoder + decoder))
            optimizer = Optim(net.reset(), MSELoss(), eps=1e-3)
            optimizer.SGD_eval(
                X_train,
                X_train,
                batch_size,
                30,
                test_size=0.33,
                return_dataframe=True,
                patience=None,
            )
            d[act_fct.__name__] = optimizer
        except KeyError:
            continue
    with open("../models/usps_dict_of_optimizer_acc_fct.pkl", "wb") as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    train_small_net()
    train_medium_net()
    # per_acc_fct()
