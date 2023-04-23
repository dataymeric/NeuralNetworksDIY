import numpy as np
from tqdm import tqdm
from icecream import ic
from .module import Module, Loss
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from copy import deepcopy


class Sequential:
    def __init__(self, *args: Module) -> None:
        self.modules = [*args]
        self.modules_copy = deepcopy(self.modules)
        self.input_list = []

    def forward(self, input):
        self.input_list = [input]
        for module in self.modules:
            # print(f"Forward de {module.__class__.__name__}")
            # print(f"Input : {input.shape}")
            input = module(input)
            self.input_list.append(input)
        # print(f"Output: {input.shape}")
        return input

    def backward(self, delta):
        """_summary_

        Parameters
        ----------
        delta : _type_
            Le delta de la Loss

        Returns
        -------
        _type_
            _description_
        """
        self.input_list.reverse()

        # print(f"Shape of loss delta : {delta.shape}")

        for i, module in enumerate(reversed(self.modules)):
            # print(f"➡️ Backward de {module.__class__.__name__}")
            # print(f"Shape of delta : {delta.shape}")
            # print(f"Shape of inputs : {self.input_list[i+1].shape}")
            module.backward_update_gradient(self.input_list[i + 1], delta)
            delta = module.backward_delta(self.input_list[i + 1], delta)
            # print(f"Backward de {module.__class__.__name__} ✅")

        # Pas sur des indices des listes !

    def update_parameters(self, eps=1e-3):
        for module in self.modules:
            # print(f"➡️ Update parameter de {module.__class__.__name__}")
            module.update_parameters(gradient_step=eps)
            module.zero_grad()

    def append(self, module: Module) -> None:
        """
        Append a module in the sequential list
        """
        raise NotImplementedError()

    def insert(self, idx: int, module: Module) -> None:
        """
        Insert a module in the sequential list
        """
        raise NotImplementedError()

    def reset(self):
        """Reset module list to the original first one and so reset all parameters.

        Returns
        -------
        Sequential
            self
        """
        self.modules = deepcopy(self.modules_copy)
        return self


class Optim:
    def __init__(self, network: Sequential, loss: Loss, eps: float) -> None:
        self.network = network
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        # y_hat = self.network.forward(batch_x).reshape(-1, 1)  # (batchsize, 1)
        # Il faut fix ce reshape, il broke en multiclass en reshapant de (batchsize=8, 2 class) => (16, 1)
        y_hat = self.network.forward(batch_x)
        loss_value = self.loss.forward(batch_y, y_hat)
        loss_delta = self.loss.backward(batch_y, y_hat)
        self.network.backward(loss_delta)
        self.network.update_parameters(self.eps)
        return loss_value

    def SGD(
        self,
        X,
        y,
        batch_size: int,
        epoch: int,
        network: Sequential = None,
        shuffle: bool = True,
    ):
        if not network:
            network = self.network

        # Shuffle ?
        if shuffle:
            shuffled_idx = np.arange(len(X))
            np.random.shuffle(shuffled_idx)
            batch_idx = np.array_split(shuffled_idx, len(X) / batch_size)
            batch_X = [X[idx] for idx in batch_idx]
            batch_Y = [y[idx] for idx in batch_idx]
        else:
            batch_X = np.array_split(X, len(X) / batch_size)
            batch_Y = np.array_split(y, len(X) / batch_size)

        loss_list = []
        for _ in tqdm(range(epoch)):
            # print(f"Epoch {i+1}\n-------------------------------")
            # for X_i, y_i in tqdm(zip(batch_X, batch_Y)):
            loss_sum = 0
            for X_i, y_i in zip(batch_X, batch_Y):
                loss_sum += self.step(X_i, y_i).sum()
            loss_list.append(loss_sum / len(y))
            # print(f"loss = {loss_list[-1]}")

        return np.array(loss_list)

    def SGD_eval(
        self,
        X,
        y,
        batch_size: int,
        epoch: int,
        test_size: float,
        network: Sequential = None,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        return_dataframe: bool = False,
    ):
        if not network:
            network = self.network

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Sauvegarde pour éventuel utilisation en dehors de la fonction
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # Batch creation
        if shuffle_train:
            shuffled_idx = np.arange(len(X_train))
            np.random.shuffle(shuffled_idx)
            batch_idx = np.array_split(shuffled_idx, len(X_train) / batch_size)
            batch_X_train = [X_train[idx] for idx in batch_idx]
            batch_Y_train = [y_train[idx] for idx in batch_idx]
        else:
            batch_X_train = np.array_split(X_train, len(X_train) / batch_size)
            batch_Y_train = np.array_split(y_train, len(X_train) / batch_size)

        if shuffle_test:
            shuffled_idx = np.arange(len(X_test))
            np.random.shuffle(shuffled_idx)
            batch_idx = np.array_split(shuffled_idx, len(X_test) / batch_size)
            batch_X_test = [X_test[idx] for idx in batch_idx]
            batch_Y_test = [y_test[idx] for idx in batch_idx]
        else:
            batch_X_test = np.array_split(X_test, len(X_test) / batch_size)
            batch_Y_test = np.array_split(y_test, len(X_test) / batch_size)

        # Training
        loss_list_train = []
        loss_list_test = []
        score_train = []
        score_test = []
        for _ in tqdm(range(epoch)):
            # print(f"Epoch {i+1}\n-------------------------------")
            # for X_i, y_i in tqdm(zip(batch_X, batch_Y)):
            loss_sum = 0
            for X_i, y_i in zip(batch_X_train, batch_Y_train):
                loss_sum += self.step(X_i, y_i).sum()
            loss_list_train.append(loss_sum / len(y_train))
            score_train.append(self.score(X_train, y_train))
            # print(f"loss = {loss_list[-1]}")

            # Epoch evaluation
            loss_sum = 0
            y_hat = self.network.forward(X_test)
            loss_list_test.append(self.loss.forward(y_test, y_hat).mean())
            score_test.append(self.score(X_test, y_test))

        if return_dataframe:
            return DataFrame(
                {
                    "epoch": [i for i in range(epoch)],
                    "loss_test": loss_list_train,
                    "loss_train": loss_list_test,
                    "score_train": score_train,
                    "score_test": score_test,
                }
            )
        else:
            return (
                np.array(loss_list_train),
                np.array(score_train),
                np.array(loss_list_test),
                np.array(score_test),
            )

    def score(self, X, y):
        assert X.shape[0] == y.shape[0], ValueError()
        if len(y.shape) != 1:  # eventual y with OneHot encoding
            y = y.argmax(axis=1)
        y_hat = np.argmax(self.network.forward(X), axis=1)
        return np.where(y == y_hat, 1, 0).mean()
