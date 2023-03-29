from base_class import Loss, Module, np


class MSELoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        return (yhat - y)**2

    def backward(self, y, yhat):
        return 2*(yhat - y)


class Linear(Module):
    def __init__(self, input:int, output:int) -> None:
        """Couche linéaire
        Son gradient $  $

        Args:
            input (int): Taille de l'entrée de la couche linéair
            output (int): Taille de la sortie de la couche linéaire
        """
        super().__init__()
        self.input = input
        self.output = output

    def zero_grad(self):
        # Annule gradient
        self._gradient = None

    def forward(self, X):
        # Calcule la passe forward
        assert X.shape[1] == self.input, ValueError(
            "X must be of shape (batch_size, input_size)")
        return X @ self._parameters

    def backward_update_gradient(self, input, delta):
        # Met a jour la valeur du gradient
        self._gradient += self.backward_delta # ?

    def backward_delta(self, input, delta):
        # Calcul la derivee de l'erreur
        # La dérivé du module * delta^h fourni par l'aval du réseau ? 
        return input * delta # ??
