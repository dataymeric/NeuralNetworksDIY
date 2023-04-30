import numpy as np
from typing import Any


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError()

    def backward(self, y, yhat):
        raise NotImplementedError()


class Module(object):
    def __init__(self):
        self._parameters = {}
        self._gradient = {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def calculate_gain(self):
        if self.__class__.__name__.lower() == "tanh":
            return 5 / 3
        elif self.__class__.__name__.lower() == "relu":
            return np.sqrt(2)
        else:
            return 1.0
        
    def zero_grad(self):
        """Réinitialise le gradient."""
        raise NotImplementedError()

    def forward(self, X):
        """Passe forward."""
        raise NotImplementedError()

    def update_parameters(self, learning_rate=1e-3):
        """Update the parameters according to the calculated gradient and the learning
        rate.
        """
        # self._parameters -= learning_rate * self._gradient
        raise NotImplementedError()

    def backward_update_gradient(self, input, delta):
        """Update gradient value given module.

        .. math::
        \frac{\partial L}{\partial w_i^h}=\sum_k \frac{\partial L}{\partial z_k^h} 
        \frac{\partial z_k^h}{\partial w_i^h}=\sum_k \delta_k^h 
        \frac{\partial z_k^h}{\partial w_i^h}, \text { let } 
        \nabla_{\mathbf{w}^h} L=\left(\begin{array}{ccc}
        \frac{\partial z_1^h}{\partial w_1^h} & \frac{\partial z_2^h}{\partial w_1^h} 
        & \cdots \\
        \frac{\partial z_1^h}{\partial w_2^h} & \ddots & \\
        \vdots & &
        \end{array}\right) \nabla_{\mathbf{z}^h L} 
        """
        raise NotImplementedError()

    def backward_delta(self, input, delta):
        """Calculates the derivative of the error and the next delta (derivative of the 
        module with respect to the to the inputs).

        .. math::
        \delta_j^{h-1}=\frac{\partial L}{\partial z_j^{h-1}}=\sum_k 
        \frac{\partial L}{\partial z_k^h} \frac{\partial z_k^h}{\partial z_j^{h-1}}, 
        \text { let } \nabla_{\mathbf{z}^{h-1}} L=\left(\begin{array}{ccc}
        \frac{\partial z_1^h}{z_1^{h-1}} & \frac{\partial z_2^h}{z_1^{h-1}} & \cdots \\
        \frac{\partial z_2^h}{z_2^{h-1}} & \ddots & \cdots \\
        \vdots &
        \end{array}\right) \nabla_{\mathbf{z}^h L}
        """
        raise NotImplementedError()
