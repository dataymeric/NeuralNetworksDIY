import numpy as np
from .module import Module

__all__ = ["TanH", "Sigmoid", "Softmax", "ReLU", "Softplus"]


class TanH(Module):
    """Hyperbolic Tangent activation function.

    .. math::
        \text{TanH}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc on ajoute rien au gradient
        Genre pas de dérivé par rapport au paramètre vu que y'a pas de paramètre
        """
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)

    def update_parameters(self, gradient_step=0.001):
        pass


class Sigmoid(Module):
    """Sigmoid activation function.

    .. math:: \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        # X a grande valeurs vas donner +inf, nécéssité de normaliser ?
        return 1 / (1 + np.exp(-X))

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc return ajoute rien au gradient"""
        pass

    def backward_delta(self, input, delta):
        # assert input.shape[1] == self.input_size, ValueError()
        # assert delta.shape == (self.input_size, self.output_size), ValueError()
        sig_X = self.forward(input)
        return delta * (sig_X * (1 - sig_X))

    def update_parameters(self, gradient_step=0.001):
        pass


class Softmax(Module):
    """Softmax activation function.

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        X_exp = np.exp(X)
        return X_exp / X_exp.sum(axis=1, keepdims=True)

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc return ajoute rien au gradient"""
        pass

    def backward_delta(self, input, delta):
        # assert input.shape[1] == self.input_size, ValueError()
        # assert delta.shape == (self.input_size, self.output_size), ValueError()
        softmax = self.forward(input)
        return delta * (softmax * (1 - softmax))

    def update_parameters(self, gradient_step=0.001):
        pass


class ReLU(Module):
    """ReLU (rectified linear unit) activation function.

    .. math:: \text{ReLU}(x) = x^+ = \max(0, x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        """math:: f'(x) = 1 \text{if} x > 0 \text{else} 0."""
        return delta * (self.forward(input) > 0)


class Softplus(Module):
    """Smooth approximation of the ReLU activation function.

    .. math:: \text{Softplus}(x) = \ln(1+e^x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.log(1 + np.exp(X))

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta / (1 + np.exp(-input))
