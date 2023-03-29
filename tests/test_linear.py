import numpy as np
from src.linear import MSELoss

batch_size = 32
d = 64 # Dim des entrées

x = np.random.random(size=(batch_size, d))
yhat = np.random.random(size=(batch_size))
y = np.random.random(size=(batch_size))

class TestMSELoss:

    def test_forward(self):
        mse = MSELoss()
        loss = mse.forward(y, yhat)
        print(loss.shape)
        assert loss.shape == (batch_size,)