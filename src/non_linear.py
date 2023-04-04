import numpy as np

from src.module import Module


class TanH(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc on ajoute rien au gradient

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (output_size, input_size) 
            _description_

        """
        pass

    def backward_delta(self, input, delta):
        """_summary_

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (output_size, input_size) 
            _description_


        Returns
        -------
        _type_
            _description_
        """
        return delta @ (1 - np.tanh(input)**2).T


class Sigmoide(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        # X a grande valeurs vas donner +inf, nécéssité de normaliser ?
        return 1 / (1 + np.exp(-X))

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc on ajoute rien au gradient

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (output_size, input_size) 
            _description_
        """
        pass

    def backward_delta(self, input, delta):
        """_summary_

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (output_size, input_size) 
            _description_


        Returns
        -------
        _type_
            _description_
        """
        return delta @ (self.forward(input) * (1 - self.forward(input))).T
