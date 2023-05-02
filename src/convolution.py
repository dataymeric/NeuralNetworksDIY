from .module import Module
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

"""
We tried to vectorize our convolutions to the maximum, prioritizing the performance.

It implies creating special views of our array, by using the `numpy.lib.stride_tricks`
functions. `sliding_window_view` is the easiest to understand, while maybe not the
fastest compared to `as_strided` (but maybe less risky too).

The calculations are done using `np.einsum`, which is relatively easy to understand
and use. The key relies in understanding the shapes of your inputs/outputs.

Shape
-----
Reminder for 1D:
input : ndarray (batch, length, chan_in)
d_out : ndarray (batch, length, chan_in) == input.shape
X_view : ndarray (batch, out_length, chan_in, self.k_size)
delta : ndarray (batch, out_length, chan_out)
_gradient["weight"] : ndarray (k_size, chan_in, chan_out)
_parameters["weight"] : ndarray (k_size, chan_in, chan_out)

Notes
-----
Notation used for `np.einsum`:
- b : batch_size
- w : width (2D) / length (1D)
- h : height (2D)
- o : out_width (2D) / out_length (1D)
- p : out_height (2D)
- c : chan_in
- d : chan_out
- k : k_size (ij for 2D)

Examples
--------
Quick demonstration of `sliding_window_view` in 1D:

>>> batch, length, chan_in, k_size = 1, 8, 1, 3
>>> input = np.random.randn(batch, length, chan_in)
>>> input
array([[[-0.41982262],
        [ 1.10111123],
        [-0.41115195],
        [ 1.18733225],
        [-1.93463567],
        [-0.22472025],
        [-0.30581971],
        [ 0.40578667]]])

>>> window = np.lib.stride_tricks.sliding_window_view(input, (1, k_size, chan_in))
>>> window
array([[[[[[-0.41982262],
           [ 1.10111123],
           [-0.41115195]]]],
        [[[[ 1.10111123],
           [-0.41115195],
           [ 1.18733225]]]],
    ...

How to deal with stride != 1?
>>> stride = 3
>>> window = np.lib.stride_tricks.sliding_window_view(input, (1, k_size, chan_in))[::1, ::stride, ::1]
>>> window
array([[[[[[-0.41982262],
           [ 1.10111123],
           [-0.41115195]]]],
        [[[[ 1.18733225],
           [-1.93463567],
           [-0.22472025]]]]]])

Then it is just a matter of reshape, to drop unnecessaries dimensions, e.g. :
>>> window = window.reshape(batch, out_length, chan_in, k_size)
>>> window
array([[[[-0.41982262,  1.10111123, -0.41115195]],
        [[ 1.18733225, -1.93463567, -0.22472025]]]])

And voilà! I will try dealing with padding in a future version, when this one is fully
operational.
"""


class Conv1D(Module):
    """1D convolution.

    Parameters
    ----------
    k_size : int
        Size of the convolving kernel.
    chan_in : int
        Number of channels in the input image.
    chan_out : in
        Number of channels produced by the convolution.
    stride : int, optional, default=1
        Stride of the convolution.
    bias : bool, optional, default=False
        If True, adds a learnable bias to the output.
    init_type : str, optional, default="normal"
        Change the initialization of parameters.

    Shape
    -----
    Input : ndarray (batch, length, chan_in)
    Output : ndarray (batch, (length - k_size) // stride + 1, chan_out)
    Weight : ndarray (k_size, chan_in, chan_out)
    Bias : ndarray (chan_out)
    """

    def __init__(
        self,
        k_size: int,
        chan_in: int,
        chan_out: int,
        stride: int = 1,
        bias: bool = False,
        init_type: str = "normal",
    ):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.include_bias = bias
        self.__init_params(init_type)

    def __init_params(self, init_type):
        gain = self.calculate_gain()

        if init_type == "normal":
            self._parameters["weight"] = np.random.randn(self.k_size, self.chan_in, self.chan_out)
            self._parameters["bias"] = np.random.randn(self.chan_out)

        elif init_type == "uniform":
            self._parameters["weight"] = np.random.uniform(0.0, 1.0, (self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.random.uniform(0.0, 1.0, (self.chan_out))

        elif init_type == "zeros":
            self._parameters["weight"] = np.zeros((self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.zeros(self.chan_out)

        elif init_type == "ones":
            self._parameters["weight"] = np.ones((self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.ones(self.chan_out)

        elif init_type == "he_normal":
            std_dev = gain * np.sqrt(2 / self.chan_in)
            self._parameters["weight"] = np.random.normal(0, std_dev, (self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.random.normal(0, std_dev, (self.chan_out))

        elif init_type == "he_uniform":
            limit = gain * np.sqrt(6 / self.chan_in)
            self._parameters["weight"] = np.random.uniform(-limit, limit, (self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.random.uniform(-limit, limit, (self.chan_out))

        elif init_type == "xavier_normal":
            std_dev = gain * np.sqrt(2 / (self.chan_in + self.chan_out))
            self._parameters["weight"] = np.random.normal(0, std_dev, (self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.random.normal(0, std_dev, (self.chan_out))

        elif init_type == "xavier_uniform":
            limit = gain * np.sqrt(6 / (self.chan_in + self.chan_out))
            self._parameters["weight"] = np.random.uniform(-limit, limit, (self.k_size, self.chan_in, self.chan_out))
            self._parameters["bias"] = np.random.uniform(-limit, limit, (self.chan_out))

        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

        if not self.include_bias:
            self._parameters["bias"] = None
            self._gradient["bias"] = None

    def zero_grad(self):
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        if self.include_bias:
            self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        assert chan_in == self.chan_in

        out_length = (length - self.k_size) // self.stride + 1

        # Prepare the input view for the convolution operation
        X_view = sliding_window_view(X, (1, self.k_size, self.chan_in))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, self.chan_in, self.k_size)

        # Perform the convolution
        self.output = np.einsum("bock, kcd -> bod", X_view, self._parameters["weight"])

        if self.include_bias:
            self.output += self._parameters["bias"]

        return self.output

    def backward_update_gradient(self, input, delta):
        batch_size, length, chan_in = input.shape
        assert chan_in == self.chan_in

        out_length = (length - self.k_size) // self.stride + 1

        X_view = sliding_window_view(input, (1, self.k_size, self.chan_in))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, self.chan_in, self.k_size)

        self._gradient["weight"] += np.einsum("bock, bod -> kcd", X_view, delta)

        if self.include_bias:
            self._gradient["bias"] += np.sum(delta, axis=(0, 1))

    def backward_delta(self, input, delta):
        _, length, chan_in = input.shape
        assert chan_in == self.chan_in

        out_length = (length - self.k_size) // self.stride + 1

        self.d_out = np.zeros_like(input)
        d_in = np.einsum("bod, kcd -> kboc", delta, self._parameters["weight"])

        for i in range(self.k_size):
            self.d_out[:, i : i + out_length * self.stride : self.stride, :] += d_in[i]

        return self.d_out

    def update_parameters(self, learning_rate):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        if self.include_bias:
            self._parameters["bias"] -= learning_rate * self._gradient["bias"]


class MaxPool1D(Module):
    """1D max pooling.

    Parameters
    ----------
    k_size : int
        Size of the convolving kernel.
    stride : int, optional, default=1
        Stride of the convolution.

    Shape
    -----
    Input : ndarray (batch, length, chan_in)
    Output : ndarray (batch, (length - k_size) // stride + 1, chan_out)
    """

    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        out_length = (length - self.k_size) // self.stride + 1

        X_view = sliding_window_view(X, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, chan_in, self.k_size)

        self.output = np.max(X_view, axis=-1)
        return self.output

    def zero_grad(self):
        pass  # No gradient in MaxPool1D

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in MaxPool1D

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        input_view = sliding_window_view(input, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        input_view = input_view.reshape(batch_size, out_length, chan_in, self.k_size)

        max_indices = np.argmax(input_view, axis=-1)

        # Create indices for batch and channel dimensions
        batch_indices, out_indices, chan_indices = np.meshgrid(
            np.arange(batch_size), np.arange(out_length), np.arange(chan_in), indexing="ij"
        )

        # Update d_out using advanced indexing
        self.d_out = np.zeros_like(input)
        self.d_out[batch_indices, out_indices * self.stride + max_indices, chan_indices] += delta[batch_indices, max_indices, chan_indices]

        return self.d_out

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in MaxPool1D


class AvgPool1D(Module):
    """1D average pooling.

    Parameters
    ----------
    k_size : int
        Size of the convolving kernel.
    stride : int, optional, default=1
        Stride of the convolution.

    Shape
    -----
    Input : ndarray (batch, length, chan_in)
    Output : ndarray (batch, (length - k_size) // stride + 1, chan_out)
    """

    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        out_length = (length - self.k_size) // self.stride + 1

        X_view = sliding_window_view(X, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, chan_in, self.k_size)

        self.output = np.mean(X_view, axis=-1)
        return self.output

    def zero_grad(self):
        pass  # No gradient in AvgPool1D

    def backward_update_gradient(self, x, delta):
        pass  # No gradient to update in AvgPool1D

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        self.d_out = np.zeros_like(input)
        delta_repeated = np.repeat(delta[:, :, np.newaxis], self.k_size, axis=2) / self.k_size

        for i in range(self.k_size):
            self.d_out[:, i : i + out_length * self.stride : self.stride] += delta_repeated[:, :, i]

        return self.d_out

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in AvgPool1D


class Flatten(Module):
    """Flatten an output.

    Shape
    -----
    Input : ndarray (batch, length, chan_in)
    Output : ndarray (batch, length * chan_in)
    """

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

    def update_parameters(self, learning_rate):
        pass
