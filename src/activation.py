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
    r"""Hyperbolic Tangent activation function.

    .. math::
        \begin{align*}
            \text{TanH}(x) &= \tanh(x)  \\
                &= \frac{\sinh x}{\cosh x} \\
                &= \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)} \\
                &= \frac{e^{2x} - 1 }{e^{2x} + 1}
            \end{align*}
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
        r"""
        .. math:: \frac{\partial M}{\partial z^h} = 1 - \tanh (z^h)^2
        """
        return delta * (1 - self(input) ** 2)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in TanH


class Sigmoid(Module):
    r"""Sigmoid activation function.

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
        r"""
        .. math :: \frac{\partial M}{\partial z^h} = \sigma(z^h) * (1 - \sigma(z^h))
        """
        sig_X = self(input)
        return delta * sig_X * (1 - sig_X)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Sigmoid


class StableSigmoid(Module):
    r"""Numerically stable Sigmoid activation function."""

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
    r"""Softmax activation function. 
    Commonly used along with a cross entropy loss. See [Softmax and cross-entropy loss](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) and [Derivative of Cross Entropy Loss with Softmax](https://www.parasdahal.com/softmax-crossentropy)

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self) -> None:
        super().__init__()

    def zero_grad(self):
        pass

    def forward(self, X):
        """
        Implemented using a log sum exp trick to avoid NaN. See [Computing softmax and numerical stability](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/).
        """
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Softmax

    def backward_delta(self, input, delta):
        r"""
        math::
            \frac{\partial M(x_i)}{\partial x_i} = M^h(x_i) * (1 - M^h(x_i))

        Plus précisement 
        math:: 
            \frac{\partial M^h(x_i)}{x_j} = \begin{cases}
                M^h(x_i) * ( 1 - M^h(x_j) ) &\text{si } i = j \\
                - M^h(x_j) M^h(x_i) &\text{ si } i \neq j \\
            \end{cases}
        """
        softmax = self(input)
        return delta * (softmax * (1 - softmax))

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Softmax


class LogSoftmax(Module):
    r"""LogSoftmax activation function.

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
    r"""ReLU (rectified linear unit) activation function.

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
        r""".. math:: \frac{\partial M}{\partial z^h} = 1 \text{ if } x > 0 \text{ else } 0."""
        return delta * (input > 0)

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in ReLU


class LeakyReLU(Module):
    r"""Leaky ReLU activation function.

    .. math::
        \text{LeakyReLU}(x) = \max(\alpha x, x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha \times x, & \text{ otherwise }
        \end{cases}
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.maximum(self.alpha * X, X)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Leaky ReLU

    def backward_delta(self, input, delta):
        r"""
        math::
            \frac{\partial M}{\partial z^h} = \begin{cases} 
                1 & \text{if } x>0, \\
                \alpha & \text{otherwise}.
            \end{cases}
        """
        dx = np.ones_like(input)
        dx[input <= 0] = self.alpha
        return delta * dx

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Leaky ReLU


class Softplus(Module):
    r"""Smooth approximation of the ReLU activation function.

    .. math:: \text{Softplus}(x) = \ln(1 + e^x)
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
        r""".. math:: \frac{\partial M}{\partial z^h} = \sigma (x) = \frac{1}{1 + e^{-x}}"""
        return delta / (1 + np.exp(-input))

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Softplus
