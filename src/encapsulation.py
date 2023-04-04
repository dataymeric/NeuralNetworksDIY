import numpy as np
from tqdm import tqdm
from src.module import Module, Loss


class Sequential():
    def __init__(self, *args: Module) -> None:
        self.modules = [*args]
        self.input_list = []

    def forward(self, input):
        self.input_list = [input]
        for module in self.modules:
            print(f"Forward de {module.__class__.__name__}")
            input = module(input)
            self.input_list.append(input)
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
        list_delta = [delta]
        for i, module in enumerate(reversed(self.modules)):
            delta = module.backward_delta(self.input_list[i], delta)
            list_delta.append(delta)
            module.backward_update_gradient(self.input_list[i], list_delta[-1])
        # Pas sur des indice des listes !

        return input

    def update_parameters(self, eps=1e-3):
        for m in self.modules:
            m.update_parameters(gradient_step=eps)
            m.zero_grad()

    def append(self, module: Module) -> None:
        raise NotImplementedError()

    def insert(self, idx: int, module: Module) -> None:
        raise NotImplementedError()


class Optim:
    def __init__(self, net: Sequential, loss: Loss, eps: float) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        y_hat = self.net.forward(batch_x)
        loss_value = self.loss.forward(batch_y, y_hat)
        loss_delta = self.loss.backward(batch_y, y_hat)
        self.net.backward(loss_delta)
        self.net.update_parameters(self.eps)
        return loss_value

    def SGD(self, X, Y, batch_size: int, epoch: int, net: Sequential = None, shuffle: bool = True):
        if not net:
            net = self.net

        # Shuffle ?
        if shuffle:
            shuffled_idx = np.arange(len(X))
            np.random.shuffle(shuffled_idx)
            batch_idx = np.array_split(shuffled_idx, batch_size)
            batch_X = [X[idx] for idx in batch_idx]
            batch_Y = [Y[idx] for idx in batch_idx]
        else:
            batch_X = np.array_split(X, batch_size)
            batch_Y = np.array_split(Y, batch_size)

        loss_list = []
        for i, X_i, y_i in tqdm(zip(range(epoch), batch_X, batch_Y)):
            loss_list.append(self.step(X_i, y_i))
            print(f'Epoch {i}, loss = {loss_list[-1]}')
        return loss_list
