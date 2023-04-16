import numpy as np
from src.module import Loss


class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        """Calcule l'erreur moyenne quadratique.

        .. math:: MSE = ||y - \hat{y}||^2

        Parameters
        ----------
        y : ndarray (batch, d,)
            Supervision.
        yhat : ndarray (batch, d,)
            Prédiction.

        Returns
        -------
        ndarray (batch,)
            Loss.
        """
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. Here it is {y.shape} and {yhat.shape}")
        return np.linalg.norm(y - yhat) ** 2

    def backward(self, y, yhat):
        """Calcule le gradient de l'erreur moyenne quadratique.

        .. math:: \nabla_{MSE} = 2(y - \hat{y})

        Parameters
        ----------
        y : ndarray (batch, d,)
            Supervision.
        yhat : ndarray (batch, d,)
            Prédiction.

        Returns
        -------
        ndarray
            Gradient.
        """
        assert y.shape == yhat.shape, ValueError(
            f"dimensionf mismatch, y and yhat must be of same dimension. Here it is {y.shape} and {yhat.shape}")
        return -2 * (y - yhat)


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. Here it is {y.shape} and {yhat.shape}")
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. Here it is {y.shape} and {yhat.shape}")
        return yhat - y


class LogCrossEntropyLoss(Loss):
    """TO DO"""

    def __init__(self) -> None:
        super().__init__()
        self.CELoss = CrossEntropyLoss()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. Here it is {y.shape} and {yhat.shape}")
        # return - yhat + np.log(np.exp(yhat).sum())

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. Here it is {y.shape} and {yhat.shape}")
        ...


class BinaryCrossEntropy(Loss):
    def __init__(self, clip=0) -> None:
        super().__init__()
        self.clip = clip
        # TO DO

    def forward(self, y, yhat):
        # Prévoir les éventuel y_hat = 0
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        return - ((y / yhat) + (1 - y) / (1 - yhat))
