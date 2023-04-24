from .module import Module
import numpy as np


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1):
        """(k_size,chan_in,chan_out)"""
        super().__init__()
        self._k_size = k_size  # taille du filtre
        self._chan_in = chan_in  # C
        self._chan_out = chan_out  # nombre de filtres
        self._stride = stride
        self._parameters = np.random.rand(k_size, chan_in, chan_out)  # filtres

    def __stride_input(self, X):
        """Calcule les fenêtres où calculer la convolution.

        Examples
        --------
        Note : ici, ce sont des arrays de taille (length, ) simplement pour donner un
        aperçu. En réalité, on fait le même processus mais sur des arrays de taille
        (batch, length, chan_in).

        >>> k_size = 3
        >>> stride = 1
        >>> input = np.arange(5)
        >>> self._stride_input(input)
        array([0, 1, 2, 3, 4])
        [[0 1 2]
         [1 2 3]
         [2 3 4]]

        >>> k_size = 2
        >>> stride = 3
        >>> self._stride_input(input)
        >>> input = np.arange(5)
        array([0, 1, 2, 3, 4])
        [[0 1]
         [3 4]]
        """
        batch_size, length, chan_in = X.shape
        assert chan_in == self._chan_in, f"X must have {self._chan_in} channels. Here "
        f"X have {chan_in} channels."
        batch_stride, length_stride, chan_stride = X.strides

        out_size = int((length - self._k_size) / self._stride + 1)
        new_shape = (batch_size, out_size, chan_in, self._k_size)
        new_strides = (
            batch_stride,
            self._stride * length_stride,
            chan_stride,
            length_stride,
        )

        return np.lib.stride_tricks.as_strided(X, new_shape, new_strides)

    def forward(self, X):
        """Performe une convolution en 1D sans boucles for.

        Parameters
        ----------
        X : ndarray (batch, length, chan_in)

        Returns
        -------
        ndarray (batch, (length-k_size)/stride + 1, chan_out)"""
        X_windows = self.__stride_input(X)
        self.inputs = X, X_windows
        output = np.einsum("blck,kcf->blf", X_windows, self._parameters)
        return output

    def backward_update_gradient(self, input, delta):
        """TO DO"""
        batch, length, chan_in = input.shape
        assert chan_in == self._chan_in
        input_windows = self.__stride_input(input)
        print(input_windows.shape)
        #self._gradient += np.einsum("blck,lcf->bkf", input_windows, delta) / batch

    def backward_delta(self, input, delta):
        """TO DO"""
        np.einsum("", delta, self._parameters)
        ...

    def forward_loops(self, X):
        """Performe une convolution en 1D avec des boucles for."""
        batch, length, chan_in = X.shape
        assert chan_in == self._chan_in

        # Initialize the output array
        out_size = int((length - self._k_size) / self._stride + 1)
        out = np.zeros((batch, out_size, self._chan_out))

        # Convolve for each batch element
        for b in range(batch):
            # Convolve for each output channel
            for c_out in range(self._chan_out):
                # Convolve for each position in the output
                for i in range(out_size):
                    # Compute the receptive field
                    start = i * self._stride
                    end = start + self._k_size

                    # Compute the convolution
                    out[b, i, c_out] = np.sum(
                        X[b, start:end, :] * self._parameters[:, :, c_out]
                    )

        return out


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        """Performe un max pooling en 1D.

        Parameters
        ----------
        X : ndarray (batch, length, chan_in)

        Returns
        -------
        (batch, (length-k_size)/stride + 1, chan_in)
        """
        batch, length, chan_in = X.shape

        out_size = int((length - self._k_size) / self._stride + 1)
        out = np.zeros((batch, out_size, chan_in))

        for b in range(batch):
            for c_in in range(chan_in):
                for i in range(out_size):
                    start = i * self._stride
                    end = start + self._k_size
                    out[b, i, c_in] = np.max(X[b, start:end, :])

        return out

    def backward_update_gradient(self, input, delta):
        """TO DO"""
        ...

    def backward_delta(self, input, delta):
        """TO DO"""
        ...


class AvgPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        """Performe un average pooling en 1D.

        Parameters
        ----------
        X : ndarray (batch, length, chan_in)

        Returns
        -------
        (batch, (length-k_size)/stride + 1, chan_in)
        """
        batch, length, chan_in = X.shape

        out_size = int((length - self._k_size) / self._stride + 1)
        out = np.zeros((batch, out_size, chan_in))

        # Faisable sans boucles ? :thinking:
        for b in range(batch):
            for c_in in range(chan_in):
                for i in range(out_size):
                    start = i * self._stride
                    end = start + self._k_size
                    out[b, i, c_in] = np.mean(X[b, start:end, :])

        return out

    def backward_update_gradient(self, input, delta):
        """TO DO"""
        ...

    def backward_delta(self, input, delta):
        """TO DO"""
        ...


class Flatten(Module):
    """(batch, length, chan_in) -> (batch, length * chan_in)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        self.batch, self.length, self.chan_in = X.shape
        return X.reshape(self.batch, self.chan_in * self.length)

    def backward(self, X):
        return X.reshape(self.batch, self.length, self.chan_in)


class Conv2D(Module):
    def __init__(self, chan_in, chan_out, k_size=4, padding=0, stride=1):
        super().__init__()
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self.__init_params()

    def __init_params(self):
        self.weights = np.zeros(
            (self._chan_out, self._chan_in, self._k_size, self._k_size),
            dtype=np.float32,
        )
        self.bias = np.zeros((self._chan_out, 1), dtype=np.float32)
        for filter in range(0, self._chan_out):
            self.weights[filter, :, :, :] = np.random.normal(
                loc=0,
                scale=(2.0 / (self._chan_in * self._k_size * self._k_size) ** 0.5),
                size=(self._chan_in, self._k_size, self._k_size),
            )

    def __stride_input(self, inputs):
        batch_size, channels, h, w = inputs.shape
        batch_stride, channel_stride, rows_stride, columns_stride = inputs.strides
        out_h = int((h - self._k_size) / self._stride + 1)
        out_w = int((w - self._k_size) / self._stride + 1)
        new_shape = (batch_size, channels, out_h, out_w, self._k_size, self._k_size)
        new_strides = (
            batch_stride,
            channel_stride,
            self._stride * rows_stride,
            self._stride * columns_stride,
            rows_stride,
            columns_stride,
        )
        return np.lib.stride_tricks.as_strided(inputs, new_shape, new_strides)

    def forward(self, inputs):
        """Accepts four dimensional input, with shape (Batch, Channels, Height, Width)"""
        input_windows = self.__stride_input(inputs)
        self.inputs = inputs, input_windows
        output = (
            np.einsum("bchwkt,fckt->bfhw", input_windows, self.weights)
            + self.bias[np.newaxis, :, np.newaxis]
        )
        return output

    def forward_loops(self, inputs):
        batch_size, in_channels, in_width, in_height = inputs.shape
        self.inputs = inputs 
        out_width = int((in_width - self.kernel_size) / self.stride + 1)
        out_height = int((in_height - self.kernel_size) / self.stride + 1)
        outputs = np.zeros((batch_size, self.num_kernels, out_width, out_height), dtype=np.float32)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for w in range(0, out_width, self.stride):
                    for h in range(0, out_height, self.stride):
                        outputs[b,k,w,h] = np.sum(
                                self.inputs[b, :, w:w+self.kernel_size, h:h+self.kernel_size] * \
                                self.weights[k, :, :, :]
                            ) + self.bias[k]
        return outputs
