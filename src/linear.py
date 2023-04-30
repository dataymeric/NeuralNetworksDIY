import numpy as np
from .module import Module


class Linear(Module):
    """Linear module.

    Parameters
    ----------
    input_size : int
        Size of input sample.
    output_size : int
        Size of output sample.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = True, init_type: str = "normal"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.include_bias = bias,
        self.__init_params(init_type)

    def __init_params(self, init_type, gain=1.0):
        if init_type == "normal":
            self._parameters["weight"] = np.random.randn(self.input_size, self.output_size)
            self._parameters["bias"] = np.random.randn(1, self.output_size)

        elif init_type == "uniform":
            self._parameters["weight"] = np.random.uniform(0.0, 1.0, (self.input_size, self.output_size))
            self._parameters["bias"] = np.random.uniform(0.0, 1.0, (1, self.output_size))

        elif init_type == "zeros":
            self._parameters["weight"] = np.zeros((self.input_size, self.output_size))
            self._parameters["bias"] = np.zeros(1, self.output_size)

        elif init_type == "ones":
            self._parameters["weight"] = np.ones((self.input_size, self.output_size))
            self._parameters["bias"] = np.ones(1, self.output_size)

        elif init_type == "he_normal":
            std_dev = gain * np.sqrt(2 / self.input_size)
            self._parameters["weight"] = np.random.normal(0, std_dev, (self.input_size, self.output_size))
            self._parameters["bias"] = np.random.normal(0, std_dev, (1, self.output_size))

        elif init_type == "he_uniform":
            limit = gain * np.sqrt(6 / self.input_size)
            self._parameters["weight"] = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
            self._parameters["bias"] = np.random.uniform(-limit, limit, (1, self.output_size))

        elif init_type == "xavier_normal":
            std_dev = gain * np.sqrt(2 / (self.input_size + self.output_size))
            self._parameters["weight"] = np.random.normal(0, std_dev, (self.input_size, self.output_size))
            self._parameters["bias"] = np.random.normal(0, std_dev, (1, self.output_size))

        elif init_type == "xavier_uniform":
            limit = gain * np.sqrt(6 / (self.input_size + self.output_size))
            self._parameters["weight"] = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
            self._parameters["bias"] = np.random.uniform(-limit, limit, (1, self.output_size))

        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

        if not self.include_bias:
            self._parameters["bias"] = None
            self._gradient["bias"] = None

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
        if self.include_bias:
            return X @ self._parameters["weight"] + self._parameters["bias"]
        else:
            return X @ self._parameters["weight"]

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size

        # Si delta : ndarray (output_size, input_size)
        self._gradient["weight"] += input.T @ delta  # (output_size, batch )
        if self.include_bias:
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
        if self.include_bias:
            self._gradient["bias"] = np.zeros((1, self.output_size))

    def update_parameters(self, learning_rate=0.001):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        if self.include_bias:
            self._parameters["bias"] -= learning_rate * self._gradient["bias"]
