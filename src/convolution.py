from .module import Module
import numpy as np


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        """(k_size,chan_in,chan_out)"""
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.parameters = np.ones((chan_out, chan_in, k_size))
        self.dw = np.zeros((chan_out, chan_in, k_size))

    def forward(self, X):
        """_summary_

        Parameters
        ----------
        X : _type_ (batch,length,chan_in)
            _description_

        Returns
        -------
        (batch, (length-k_size)/stride +1,chan_out)
        """
        pass


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """(batch, length, chan_in) -> (batch,(length-k_size)/stride +1,chan_in)."""
        pass


class Flatten(Module):
    """(batch, length, chan_in) -> (batch, length * chan_in)"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return X.reshape(self.batch, self.chan_in * self.length)

    def backward(self, X):
        return X.reshape(self.batch, self.length, self.chan_in)


class ReLU(Module):
    """ReLU (rectified linear unit) activation function."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def backward(self, input, delta):
        return delta * (self.forward(input) > 0)
