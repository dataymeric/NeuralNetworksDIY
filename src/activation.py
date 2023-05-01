import numpy as np
from .module import Module

__all__ = [
    "TanH",
    "Sigmoid",
    "StableSigmoid",
    "Softmax",
    "LogSoftmax",
    "ReLU",
    "LeakyReLU",
    "Softplus",
]


class TanH(Module):
    """Hyperbolic Tangent activation function.

    .. math::
        \text{TanH}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in TanH

    def backward_delta(self, input, delta):
        return delta * (1 - self(input) ** 2)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in TanH


class Sigmoid(Module):
    """Sigmoid activation function.

    .. math:: \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        # X a grande valeurs vas donner +inf, nécéssité de normaliser ?
        return 1 / (1 + np.exp(-X))

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Sigmoid

    def backward_delta(self, input, delta):
        sig_X = self(input)
        return delta * sig_X * (1 - sig_X)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Sigmoid


class StableSigmoid(Module):
    """Numerically stable Sigmoid activation function."""

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Sigmoid

    def backward_delta(self, input, delta):
        sig_X = self(input)
        return delta * sig_X * (1 - sig_X)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Sigmoid


class Softmax(Module):
    """Softmax activation function.

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))  # log-sum-exp trick
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Softmax

    def backward_delta(self, input, delta):
        softmax = self(input)
        return delta * (softmax * (1 - softmax))

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Softmax


class LogSoftmax(Module):
    """LogSoftmax activation function.

    .. math::
            \text{LogSoftmax}(x_{i}) =
            \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        X_shifted = X - np.max(X, axis=-1, keepdims=True)
        return X_shifted - np.log(np.sum(np.exp(X_shifted), axis=-1, keepdims=True))

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in LogSoftmax

    def backward_delta(self, input, delta):
        softmax = np.exp(self(input))
        return delta - softmax * np.sum(delta, axis=-1, keepdims=True)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in LogSoftmax


class ReLU(Module):
    """ReLU (rectified linear unit) activation function.

    .. math:: \text{ReLU}(x) = x^+ = \max(0, x)
    """

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.maximum(0, X)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in ReLU

    def backward_delta(self, input, delta):
        """math:: f'(x) = 1 \text{if} x > 0 \text{else} 0."""
        return delta * (input > 0)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in ReLU


class LeakyReLU(Module):
    """Leaky ReLU activation function.
    
    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.where(X >= 0, X, self.alpha * X)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Leaky ReLU

    def backward_delta(self, input, delta):
        dx = np.ones_like(input)
        dx[input <= 0] = self.alpha
        return delta * dx

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Leaky ReLU


class Softplus(Module):
    """Smooth approximation of the ReLU activation function.

    .. math:: \text{Softplus}(x) = \ln(1+e^x)
    """

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.log(1 + np.exp(X))

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Softplus

    def backward_delta(self, input, delta):
        return delta / (1 + np.exp(-input))

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Softplus
