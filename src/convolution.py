from .module import Module
import numpy as np


class Conv1D(Module):
    """1D convolution.

    Parameters
    ----------
    X : ndarray (batch, length, chan_in)

    Returns
    -------
    ndarray (batch, (length-k_size)/stride + 1, chan_out)
    """

    def __init__(self, k_size, chan_in, chan_out, stride):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters["weight"] = np.random.randn(k_size, chan_in, chan_out)
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])

    def zero_grad(self):
        return np.zeros_like(self._parameters["weight"])

    def forward(self, X):
        batch_size, length, _ = X.shape
        self.output_shape = (batch_size, (length - self.k_size) // self.stride + 1, self.chan_out)

        # Prepare the input view for the convolution operation
        x_view = np.lib.stride_tricks.sliding_window_view(X, (1, self.k_size, self.chan_in))[::1, :: self.stride, ::1]
        x_view = x_view.reshape(batch_size, self.output_shape[1], self.k_size * self.chan_in)

        # Perform the convolution
        self.output = np.einsum("ijk, lk -> ijl", x_view, self._parameters["weight"].reshape(self.chan_out, -1))
        return self.output

    def backward_update_gradient(self, x, delta):
        batch, length, chan_in = x.shape
        out_length = (length - self.k_size) // self.stride + 1
        x_view = np.lib.stride_tricks.sliding_window_view(x, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        x_view = x_view.reshape(batch, out_length, chan_in, self.k_size)

        for k in range(self.chan_out):
            for i in range(self.chan_in):
                for j in range(self.k_size):
                    self._gradient["weight"][j, i, k] += np.sum(x_view[:, :, i, j] * delta[:, :, k])

        # self._gradient["bias"] += np.sum(delta, axis=(0, 1))

    def backward_delta(self, input, delta):
        batch, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1
        d_out = np.zeros_like(input)

        for i in range(self.k_size):
            for k in range(self.chan_out):
                d_out[:, i : i + out_length * self.stride : self.stride, :] += np.outer(delta[:, :, k], self._parameters["weight"][i, :, k]).reshape(batch, out_length, chan_in)

        return d_out

    def update_parameters(self, learning_rate):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]


class MaxPool1D(Module):
    """1D max pooling.

    Parameters
    ----------
    X : ndarray (batch, length, chan_in)

    Returns
    -------
    ndarray (batch, (length-k_size)/stride + 1, chan_in)
    """

    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, x):
        batch, length, chan_in = x.shape
        out_length = (length - self.k_size) // self.stride + 1
        x_view = np.lib.stride_tricks.sliding_window_view(x, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        x_view = x_view.reshape(batch, out_length, chan_in, -1)
        output = np.max(x_view, axis=-1)
        return output

    def backward_update_gradient(self, x, delta):
        pass  # No gradient to update in MaxPool1D

    def backward_delta(self, x, delta):
        batch, length, chan_in = x.shape
        d_out = np.zeros_like(x)

        for b in range(batch):
            for i in range(0, length - self.k_size + 1, self.stride):
                window = x[b, i : i + self.k_size]
                max_indices = np.argmax(window, axis=0)
                d_out[b, i + max_indices] += delta[b, i // self.stride]

        return d_out

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in MaxPool1D


class AvgPool1D(Module):
    """1D average pooling.

    Parameters
    ----------
    X : ndarray (batch, length, chan_in)

    Returns
    -------
    ndarray (batch, (length-k_size)/stride + 1, chan_in)
    """

    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, x):
        batch, length, chan_in = x.shape
        out_length = (length - self.k_size) // self.stride + 1
        x_view = np.lib.stride_tricks.sliding_window_view(x, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        x_view = x_view.reshape(batch, out_length, chan_in, -1)
        output = np.mean(x_view, axis=-1)
        return output

    def backward_update_gradient(self, x, delta):
        pass  # No gradient to update in AvgPool1D

    def backward_delta(self, x, delta):
        batch, length, chan_in = x.shape
        out_length = (length - self.k_size) // self.stride + 1
        d_out = np.zeros_like(x)
        delta_repeated = np.repeat(delta[:, :, np.newaxis], self.k_size, axis=2) / self.k_size
        for i in range(self.k_size):
            d_out[:, i : i + out_length * self.stride : self.stride] += delta_repeated[:, :, i]
        return d_out

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in AvgPool1D


class Flatten(Module):
    """Flatten an output.

    Parameters
    ----------
    X : ndarray (batch, length, chan_in)

    Returns
    -------
    ndarray (batch, length * chan_in)
    """

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, learning_rate):
        pass
