import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super(LinearLayer, self).__init__()
       
        stdv = 1. / np.sqrt(in_features)
        self.parameters.append(np.random.uniform(-stdv, stdv, size=(out_features, in_features)))
        self.parameters.append(np.random.uniform(-stdv, stdv, size=out_features))
        
        self.parameters_grads.append(np.zeros_like(self.parameters[0]))
        self.parameters_grads.append(np.zeros_like(self.parameters[1]))

        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input @ self.parameters[0].T + self.parameters[1]
        self.input = input
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        grad_input = output_grad @ self.parameters[0]
        dW = np.matmul(np.swapaxes(output_grad, -1, -2), self.input)
        while len(dW.shape) > 2:
            dW = np.sum(dW, axis=0)
        self.parameters_grads[0] = dW
        db = np.sum(output_grad, axis=0)
        while len(db.shape) > 1:
            db = np.sum(db, axis=0)
        self.parameters_grads[1] = db

        return grad_input
