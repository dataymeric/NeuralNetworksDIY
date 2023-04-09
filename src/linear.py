import numpy as np
from src.module import Loss, Module


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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        return - ( (y / yhat) + (1 - y) / (1 - yhat) )


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, param_init: str = None) -> None:
        """Couche linéaire
        self._parameters : (input, output)

        Args:
            input (int): Taille de l'entrée de la couche linéair
            output (int): Taille de la sortie de la couche linéaire
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Initialisation des paramètres
        # self._parameters = np.random.random(size=(input_size, output))
        # self._parameters = np.random.randn(input_size, output_size)
        # self._parameters = np.ones(shape=(input_size, output_size))
        # self._bias = np.ones(shape=(1, output_size))

        self._parameters = np.random.normal(
            0, 1, (input_size, output_size)) * np.sqrt(2 / (input_size + output_size))
        self._bias = np.random.normal(
            0, 1, (1, output_size)) * np.sqrt(2 / (input_size + output_size))

        # Initialisation du gradient
        self._gradient = np.zeros((input_size, output_size))
        self._gradient_bias = np.zeros((1, output_size))

    def forward(self, X):
        """X@w 
        (batch, input_size) @ (input_size, output_size) = (batch, output_size)

        Parameters
        ----------
        X : ndarray (batch, input_size)
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert X.shape[1] == self.input_size, ValueError(
            "X must be of shape (batch_size, input_size)")
        return X @ self._parameters + self._bias

    def backward_update_gradient(self, input, delta):
        """_summary_

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (input_size, output_size) 
            _description_
        """
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # Si delta : ndarray (output_size, input_size)
        self._gradient += input.T @ delta  # (output_size, batch )
        self._gradient_bias += delta.sum(axis=0)
        # Plutot logique avec l'idée que le la Loss : R^? ==> R donne un gradient de cette forme

        # Si delta : ndarray (batch, output_size, input_size)
        # self._gradient += delta @ input.T # (batch, output_size, batch)
        # Un peu étrange quoi

    def backward_delta(self, input, delta):
        """_summary_

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (input_size, output_size)
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # c'est la dérivé du module par rapport aux entrée !!!
        # delta * self._parameters

        # Si delta : ndarray (output, input_size)
        return (delta @ self._parameters.T)

        # Si delta : ndarray (batch, output_size, input_size) 🤔
        # return np.repeat ..... (delta @ self._parameters.T)

    def zero_grad(self):
        self._gradient = np.zeros((self.input_size, self.output_size))
        self._gradient_bias = np.zeros((1, self.output_size))

    def update_parameters(self, gradient_step=0.001):
        self._bias -= gradient_step * self._gradient_bias
        return super().update_parameters(gradient_step)
