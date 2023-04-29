import numpy as np
from .module import Module


class Linear(Module):
    """Module linéaire.

    input_size : int
        Taille de l'entrée de la couche linéaire.
    output_size : int
        Taille de la sortie de la couche linéaire.
    _parameters : (input_size, output_size)

    """

    def __init__(self, input_size: int, output_size: int, param_init: str = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.__init_params()

    def __init_params(self):
        # Initialisation des paramètres
        # self._parameters["weight"] = np.random.random(size=(input_size, output))
        # self._parameters["weight"] = np.random.randn(input_size, output_size)
        # self._parameters["weight"] = np.ones(shape=(input_size, output_size))
        # self._parameters["bias"] = np.ones(shape=(1, output_size))

        # Initialisation des paramètres
        self._parameters["weight"] = np.random.normal(
            0, 1, (self.input_size, self.output_size)
        ) * np.sqrt(2 / (self.input_size + self.output_size))
        self._parameters["bias"] = np.random.normal(
            0, 1, (1, self.output_size)
        ) * np.sqrt(2 / (self.input_size + self.output_size))

        # Initialisation du gradient
        self._gradient["weight"] = np.zeros((self.input_size, self.output_size))
        self._gradient["bias"] = np.zeros((1, self.output_size))

    def forward(self, X):
        """X @ w
        (batch, input_size) @ (input_size, output_size) = (batch, output_size)

        Parameters
        ----------
        X : ndarray (batch, input_size)

        Returns
        -------
        ndarray (batch, output_size)
        """
        assert X.shape[1] == self.input_size, ValueError(
            "X must be of shape (batch_size, input_size)"
        )
        return X @ self._parameters["weight"] + self._parameters["bias"]

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # Si delta : ndarray (output_size, input_size)
        self._gradient["weight"] += input.T @ delta  # (output_size, batch )
        self._gradient["bias"] += delta.sum(axis=0)
        # Plutot logique avec l'idée que le la Loss : R^? ==> R donne un gradient de cette forme

        # Si delta : ndarray (batch, output_size, input_size)
        # self._gradient += delta @ input.T # (batch, output_size, batch)
        # Un peu étrange quoi

    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # c'est la dérivé du module par rapport aux entrée !!!
        # delta * self._parameters

        # Si delta : ndarray (output, input_size)
        return delta @ self._parameters["weight"].T

        # Si delta : ndarray (batch, output_size, input_size) 🤔
        # return np.repeat ..... (delta @ self._parameters.T)

    def zero_grad(self):
        self._gradient["weight"] = np.zeros((self.input_size, self.output_size))
        self._gradient["bias"] = np.zeros((1, self.output_size))

    def update_parameters(self, learning_rate=0.001):
        self._parameters["bias"] -= learning_rate * self._gradient["bias"]
