from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.linear import Linear
from src.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropy, LogCrossEntropyLoss
from src.non_linear import TanH, Sigmoide, SoftMax
from src.encapsulation import Sequential, Optim
np.random.seed(42)


batch_size = 8

X, y = load_digits(return_X_y=True, n_class=2)
y_oh = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

net = Sequential(
    Linear(64, 32),
    TanH(),
    Linear(32, 16),
    TanH(),
    Linear(16, 8),
    TanH(),
    Linear(8, 2),
    Sigmoide(),
    # SoftMax(),
)

optimizer = Optim(net, CrossEntropyLoss(), eps=1e-2)
lossList = optimizer.SGD(X, y_oh, batch_size, 10)
print(lossList)
print(optimizer.score(X, y))
