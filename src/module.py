import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Réinitialise à 0 le gradient."""
        self._gradient = 0

    def forward(self, X):
        # Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        # Calcule la mise a jour des paramètres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        # Met a jour la valeur du gradient
        # C'est la somme dans le sujet
        # EQUATION 1
        pass 

    def backward_delta(self, input, delta):
        # Calcul la dérivée de l'erreur
        # calcul le prochain delta
        # EQUATION 2
        pass
