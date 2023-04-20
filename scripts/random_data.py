import numpy as np
import pandas as pd
from src.linear import Linear
from src.loss import MSELoss, CrossEntropyLoss, BCELoss, LogSoftmax
from src.activation import TanH, Sigmoide, Softmax
from src.encapsulation import Sequential, Optim
np.random.seed(42)

batch_size = 32
d = 2  # Dim des entrées

X = np.random.random(size=(256, d))
y = np.random.choice([1], size=(256, 1))

net = Sequential(
    Linear(2, 2),
    TanH(),
    Linear(2, 1),
    Sigmoide(),
)

optimizer = Optim(net, CrossEntropyLoss(), eps=1e-1)
lossList = optimizer.SGD(X, y, batch_size, 10)
print(lossList)
pd.Series(lossList).plot()
