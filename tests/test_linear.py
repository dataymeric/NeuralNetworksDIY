import numpy as np
from src.linear import MSELoss, Linear

batch_size = 16
d = 64 # Dim des entrées
output_size = 32
mse = MSELoss()
linear = Linear(d, output_size)
linear._parameters = np.ones(shape=(d, output_size))

X = np.random.random(size=(batch_size, d))
yhat = np.random.random(size=(batch_size))
y = np.random.random(size=(batch_size))

class TestMSELoss:

    def test_forward(self):
        loss = mse.forward(y, yhat)
        print(loss.shape)
        assert loss.shape == (batch_size,)
        
    def test_backward(self):
        """A vérifier 
        """
        pass

class TestLinear:
    
    def test_forward(self):
        assert linear.forward(X).shape == (batch_size, output_size)
    
