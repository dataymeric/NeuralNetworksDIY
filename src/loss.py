import numpy as np
from .module import Loss


__all__ = ["MSELoss", "CrossEntropyLoss", "BCELoss", "CELogSoftmax"]


class MSELoss(Loss):
    """Mean Squared Error loss function.

    .. math:: MSE = ||y - \hat{y}||^2
    .. math:: \nabla_{MSE} = 2(y - \hat{y})
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
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return yhat - y


class BCELoss(Loss):
    """Binary Cross Entropy loss function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        yhat += 1e-10
        return -(
            y * np.maximum(-100, np.log(yhat))
            + (1 - y) * np.maximum(-100, np.log(1 - yhat))
        )

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        yhat += 1e-10
        return -((y / yhat) + (1 - y) / (1 - yhat))


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
