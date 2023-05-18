import os
import sys
sys.path.append("../")
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from src.linear import Linear
from src.loss import *
from src.activation import TanH, Sigmoid, StableSigmoid, Softmax, LogSoftmax, ReLU, TanH, Softplus
from src.encapsulation import Sequential, Optim

np.random.seed(42)

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Data Loading                                                           │
    └────────────────────────────────────────────────────────────────────────┘
 """

def load_data(rootdir, class_list=None, trim: float = 0.5):
    def normalize_batch_image(X):
        mn = np.min(X)
        mx = np.max(X)
        X_norm = (X - mn) * (1.0 / (mx - mn))
        return X_norm

    train = pd.read_csv(os.path.join(rootdir, "fashion-mnist_train.csv"))
    # Filtering requested class
    if class_list:
        train = train[train["label"].isin(class_list)]
    y_train = train["label"].values
    X_train = train.drop(columns="label").values

    test = pd.read_csv(os.path.join(rootdir, "fashion-mnist_test.csv"))
    # Filtering requested class
    if class_list:
        test = test[test["label"].isin(class_list)]
    y_test = test["label"].values
    X_test = test.drop(columns="label").values

    trim_train = int(len(X_train) * trim)
    # trim_test = int(len(X_test) * trim)
    trim_test = int(len(X_test))

    # Normalization + trimming
    X_train = normalize_batch_image(X_train[:trim_train, :])
    X_test = normalize_batch_image(X_test[:trim_test, :])
    y_train = y_train[:trim_train]
    y_test = y_test[:trim_test]

    return (X_train, X_test, y_train, y_test)


rootdir = "../data/fashion-mnist/"
X_train, X_test, y_train, y_test = load_data(rootdir, trim=0.5)

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
        Linear(784, 64),
        TanH(),
    ]
    decoder = [
        Linear(64, 784),
        Sigmoid()
    ]
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
    with open("../models/fashion-mnist_30_epoch_simple_net.pkl", "wb") as f:
        pickle.dump(optimizer, f)

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Training medium network                                                │
    └────────────────────────────────────────────────────────────────────────┘
"""
def train_medium_net():
    encoder = [
        Linear(784, 256),
        TanH(),
        Linear(256, 64),
        TanH(),
    ]
    decoder = [
        Linear(64, 256),
        TanH(),
        Linear(256, 784),
        Sigmoid()
    ]
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
    with open("../models/fashion-mnist_30_epoch_medium_net.pkl", "wb") as f:
        pickle.dump(optimizer, f)

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Training big network                                                   │
    └────────────────────────────────────────────────────────────────────────┘
 """
def train_big_net():
    encoder = [
        Linear(784, 512),
        TanH(),
        Linear(512, 256),
        TanH(),
        Linear(256, 128),
        TanH(),
        Linear(128, 64),
        TanH(),
    ]
    decoder = [
        Linear(64, 128),
        TanH(),
        Linear(128, 256),
        TanH(),
        Linear(256, 512),
        TanH(),
        Linear(512, 784),
        Sigmoid(),
    ]

    net_cplx_fort = Sequential(*(encoder + decoder))
    optimizer = Optim(net_cplx_fort.reset(), BCELoss(), eps=1e-4)
    optimizer.SGD_eval(
        X_train,
        X_train,
        batch_size,
        30,
        test_size=0.33,
        return_dataframe=True,
        patience=None,
    )
    with open("../models/fashion-mnist_30_epoch_big_net.pkl", "wb") as f:
        pickle.dump(optimizer, f)

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Small network with different activation function                       │
    └────────────────────────────────────────────────────────────────────────┘
 """

def per_acc_fct():
    d = {}
    for act_fct in [TanH, Sigmoid, StableSigmoid, Softmax, LogSoftmax, ReLU, TanH, Softplus]:
        try:
            encoder = [
                Linear(784, 64),
                act_fct(),
            ]
            decoder = [
                Linear(64, 784),
                Sigmoid()
            ]
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
    with open("../models/fashion-mnist_dict_of_optimizer_acc_fct.pkl", "wb") as f:
        pickle.dump(d, f)

if __name__ == '__main__':
    # train_small_net()
    # train_medium_net()
    # train_big_net()
    per_acc_fct()