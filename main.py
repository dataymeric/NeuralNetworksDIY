from src.linear import MSELoss, Linear
from module import Module
import numpy as np
from tqdm import tqdm

lin = Linear(1, 1)
# non_linear = 
L = MSELoss()

class Sequential:
    def __init__(self, *args: Module) -> None:
        self.modules = [*args]

    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input

class FirstModel(Module):
    pass