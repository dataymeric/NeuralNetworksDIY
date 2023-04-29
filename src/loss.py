import numpy as np
from .module import Loss


__all__ = ["MSELoss", "CrossEntropyLoss", "BCELoss", "CELogSoftmax"]


class MSELoss(Loss):
    """Mean Squared Error loss function.

    .. math:: MSE = ||y - \hat{y}||^2
    .. math:: \nabla_{MSE} = -2(y - \hat{y})
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return np.linalg.norm(y - yhat) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return -2 * (y - yhat)


class CrossEntropyLoss(Loss):
    """Cross Entropy loss function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return -np.sum(y * yhat) / y.shape[0]

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return -y / y.shape[0]


class BCELoss(Loss):
    """Binary Cross Entropy loss function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return -np.mean(y * np.log(np.clip(yhat, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - yhat, 1e-10, 1)))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return -(y / np.clip(yhat, 1e-10, 1) - (1 - y) / np.clip(1 - yhat, 1e-10, 1))


class CELogSoftmax(Loss):
    """TO DO

    .. math::
        \text{CE}(y, \hat{y}) = - \log \frac {e^{\hat{y}_y}} {\sum_{i=1}^{K}
        e^{\hat{y}_i}} = -\hat{y}_y} + \log \sum_{i=1}^{K}e^{\hat{y}_i}}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        # return - yhat + np.log(np.exp(yhat).sum())
        return np.log(np.exp(yhat).sum(axis=1)) - (y * yhat).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape((-1, 1)) - y
