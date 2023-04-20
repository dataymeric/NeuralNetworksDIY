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
        assert chan_in == self._chan_in

        # Initialize the output array
        out_size = int(np.floor((length - self._k_size) / self._stride) + 1)
        out = np.zeros((batch, out_size, self._chan_out))

        # Convolve for each batch element
        # Faisable sans boucles ? :thinking:
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
    def __init__(self, k_size, stridede .
                 ):
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

        out_size = int(np.floor((length - self._k_size) / self._stride) + 1)
        out = np.zeros((batch, out_size, chan_in))

        # Faisable sans boucles ? :thinking:
        for b in range(batch):
            for c_in in range(chan_in):
                for i in range(out_size):
                    start = i * self._stride
                    end = start + self._k_size
                    out[b, i, c_in] = np.max(X[b, start:end, :])

        return out


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

        out_size = int(np.floor((length - self._k_size) / self._stride) + 1)
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
        ...

    def backward_delta(self, input, delta):
        ...

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
        """math:: f(x) = x^+"""
        return np.maximum(0, X)

    def backward(self, input, delta):
        """math:: f'(x) = 1 \text{if} x > 0 \text{else} 0."""
        return delta * (self.forward(input) > 0)


class SoftPlus(Module):
    """Smooth approximation of the ReLU activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        """math:: $f(x) = \ln(1+e^x)$

        Possibilité d'ajouter un hyperparamètre :math:`\beta`. Alors :

        .. math:: f(x) = \frac{1}{\beta}\ln(1+e^{\beta x})
        """
        return np.log(1 + np.exp(X))

    def backward(self, input, delta):
        """math:: f'(x) = \frac{1}{1+e^{-x}}

        Avec hyperparamètre :math:`\beta` :
        .. math:: f(x) = \frac{\beta}{1+e^{-\beta x}}
        """
        return delta / (1 + np.exp(-input))
