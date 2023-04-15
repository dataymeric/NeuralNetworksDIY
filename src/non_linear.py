import numpy as np

from .module import Module


class TanH(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc on ajoute rien au gradient
        Genre pas de dérivé par rapport au paramètre vu que y'a pas de paramètre

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (input_size, output_size) 
            _description_

        """
        pass

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
        return delta * (1 - np.tanh(input)**2)

    def update_parameters(self, gradient_step=0.001):
        ...

class Sigmoide(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        # X a grande valeurs vas donner +inf, nécéssité de normaliser ?
        return 1 / (1 + np.exp(-X))

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc return ajoute rien au gradient

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (input_size, output_size) 
            _description_
        """
        pass

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
        # assert input.shape[1] == self.input_size, ValueError()
        # assert delta.shape == (self.input_size, self.output_size), ValueError()
        sig_X = self.forward(input)
        return delta * (sig_X * (1 - sig_X))

    def update_parameters(self, gradient_step=0.001):
        ...
    
class SoftMax(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        X_exp = np.exp(X)
        return X_exp / X_exp.sum(axis=1, keepdims=True)

    def backward_update_gradient(self, input, delta):
        """Rien à apprendre donc return ajoute rien au gradient

        Parameters
        ----------
        input : ndarray (batch, input_size)
            _description_
        delta : ndarray (input_size, output_size) 
            _description_
        """
        pass

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
        # assert input.shape[1] == self.input_size, ValueError()
        # assert delta.shape == (self.input_size, self.output_size), ValueError()
        softmax = self.forward(input)
        return delta * (softmax * (1 - softmax))
        
    def update_parameters(self, gradient_step=0.001):
        ...