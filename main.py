import numpy as np
from src.linear import MSELoss, Linear
from src.non_linear import TanH, Sigmoide
from src.encapsulation import Sequential, Optim

batch_size = 8
d = 2  # Dim des entrées

X = np.random.random(size=(256, d))
y = np.random.choice([-1, 1], size=(256, 1))

net = Sequential(
    Linear(2, 2),
    TanH(),
    Linear(2, 1),
    TanH(),
    Sigmoide(),
)

optimizer = Optim(net, MSELoss(), eps=1e-4)
optimizer.SGD(X, y, batch_size, 3)
