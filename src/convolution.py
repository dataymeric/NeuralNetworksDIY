from .module import Module
import numpy as np


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1):
        """(k_size,chan_in,chan_out)"""
        super().__init__()
        self.k_size = k_size  # taille du filtre
        self.chan_in = chan_in  # C
        self.chan_out = chan_out  # nombre de filtres
        self.stride = stride
        self.parameters = np.ones((k_size, chan_in, chan_out))  # filtres

    def forward(self, X):
        """Performe une convolution en 1D.

        Parameters
        ----------
        X : ndarray (batch,length,chan_in)

        Returns
        -------
        ndarray (batch, (length-k_size)/stride + 1, chan_out)
        """
        batch, length, chan_in = X.shape

        # Initialize the output array
        out_size = int(np.floor((length - self.k_size) / self.stride) + 1)
        out = np.zeros((batch, out_size, self.chan_out))

        # Convolve for each batch element
        # Faisable sans boucles ? :thinking:
        for b in range(batch):
            # Convolve for each output channel
            for c_out in range(self.chan_out):
                # Convolve for each position in the output
                for i in range(out_size):
                    # Compute the receptive field
                    start = i * self.stride
                    end = start + self.k_size

                    # Compute the convolution
                    out[b, i, c_out] = np.sum(
                        X[b, start:end, :] * self.parameters[:, :, c_out]
                    )

        return out


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """(batch, length, chan_in) -> (batch, (length-k_size)/stride + 1, chan_in)."""
        batch, length, chan_in = X.shape

        out_size = int(np.floor((length - self.k_size) / self.stride) + 1)
        out = np.zeros((batch, out_size, self.chan_out))

        # Faisable sans boucles ? :thinking:
        for b in range(batch):
            for c_out in range(self.chan_out):
                for i in range(out_size):
                    start = i * self.stride
                    end = start + self.k_size
                    out[b, i, c_out] = np.max(X[b, start:end, :])

        return out


class Flatten(Module):
    """(batch, length, chan_in) -> (batch, length * chan_in)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        batch, length, chan_in = X.shape
        return X.reshape(batch, chan_in * length)

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
