import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super(ReluLayer, self).__init__()
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output = np.maximum(input, 0)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        grad_input = np.multiply(output_grad, self.input > 0)
        return grad_input

