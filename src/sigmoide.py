from module import Module, np
import numpy as np

class TanH(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.tanh(X)
    
    def backward_delta(self, input, delta):
        return
    
    def backward_update_gradient(self, input, delta):
        return 

class Sigmoide(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    def backward_delta(self, input, delta):
        return
