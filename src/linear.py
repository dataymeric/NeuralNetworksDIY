from base_class import Loss, Module, np


class MSELoss(Loss):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y, yhat):
        return (yhat - y)**2

    def backward(self, y, yhat):
        pass
    
