import numpy as np
from tqdm import tqdm
from icecream import ic
from .module import Module, Loss


class Sequential:
    def __init__(self, *args: Module) -> None:
        self.modules = [*args]
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
        Y,
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
            batch_Y = [Y[idx] for idx in batch_idx]
        else:
            batch_X = np.array_split(X, len(X) / batch_size)
            batch_Y = np.array_split(Y, len(X) / batch_size)

        loss_list = []
        for _ in tqdm(range(epoch)):
            # print(f"Epoch {i+1}\n-------------------------------")
            # for X_i, y_i in tqdm(zip(batch_X, batch_Y)):
            for X_i, y_i in zip(batch_X, batch_Y):
                last_loss = self.step(X_i, y_i)
            loss_list.append(np.mean(last_loss))
            # print(f"loss = {loss_list[-1]}")

        return np.array(loss_list)

    def score(self, X, y):
        y_hat = np.argmax(self.network.forward(X), axis=1)
        return np.where(y == y_hat, 1, 0).mean()
