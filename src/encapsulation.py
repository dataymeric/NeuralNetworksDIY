from module import Module

class Sequential():
    def __init__(self, *args: Module) -> None:
        self.modules = [*args]
        self.input_list = []

    def forward(self, input):
        self.input_list = [input]
        for module in self.modules:
            input = module(input)
            self.input_list.append(input)
        return input

    def backward(self):
        list_delta = []
        for i, module in enumerate(reversed(self.modules)):
            delta = module.backward_delta(self.input_list[i], delta)
            list_delta.append(delta)
            module.backward_update_gradient(self.input_list[i], list_delta[-1])
        # Pas sur des indice des listes ! 
        
        return input
    
    def update_parameters(self, eps = 1e-3):
        for m in self._modules:
            m.update_parameters(gradient_step=eps)
            m.zero_grad()
            
    def append(self, module: Module) -> None:
        raise NotImplementedError()
    
    def insert(self, idx: int, module: Module) -> None:
        raise NotImplementedError()