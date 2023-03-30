from module import Loss, Module, np


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
        assert y.shape == yhat.shape, ValueError("dimension mismatch, y and yhat must of same dimension.")
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
        assert y.shape == yhat.shape, ValueError("dimension mismatch, y and yhat must be of same dimension.")
        return 2 * (y - yhat)


class Linear(Module):
    def __init__(self, input: int, output: int) -> None:
        """Couche linéaire
        Son gradient $  $

        Args:
            input (int): Taille de l'entrée de la couche linéair
            output (int): Taille de la sortie de la couche linéaire
        """
        super().__init__()
        self.input = input
        self.output = output

    def forward(self, X):
        # Calcule la passe forward
        assert X.shape[1] == self.input, ValueError(
            "X must be of shape (batch_size, input_size)")
        return X @ self._parameters
    
    def backward_update_gradient(self, input, delta):
        self._gradient += delta * input
    
    def backward_delta(self, input, delta):
        """_summary_

        Parameters
        ----------
        input : ndarray (batch, d)
            _description_
        delta : ndarray (input, output)
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # c'est la dérivé du module par rapport aux entrée !!!
        # delta * self._parameters
        return (delta * input).sum(axis=1) # (1 ,d)
