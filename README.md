# Neural Networks Library

This project is an implementation of a Neural Networks library from scratch, using only Python and Numpy. It is inspired by the old Lua versions of [Torch](https://en.wikipedia.org/wiki/Torch_(machine_learning)), before the introduction of autograd.

## Features

- Implementation of essential modules such as linear layers, 1D convolutions, and more.
- Continuous integration and deployment of [documentation](https://dataymeric.github.io/NeuralNetworksDIY/) with mathematical explanations.
- Efficient computation by avoiding for loops through advanced use of Numpy.
- Clean and well-structured code.
- A [detailed report](reports/report.pdf) showcasing various examples and experiments using different architectures (available in the `scripts` and `notebooks` folders).

## Documentation

The documentation for this project is generated using Sphinx and is available [here](https://dataymeric.github.io/NeuralNetworksDIY/). It includes detailed explanations of the implemented modules, usage examples, and mathematical foundations.

## Installation

To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
To use the library, simply import the necessary modules from the `src` directory. For example:
```py
from src.activation import Sigmoid, ReLU
from src.linear import Linear
from src.loss import BCELoss, MSELoss
```

## Authors
Made with ❤️ by @dataymeric & @CharlesAttend during our first years of Master DAC at Sorbonne University.

## License
This project is licensed under the MIT License - see the LICENSE file for details.