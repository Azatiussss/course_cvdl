import numpy as np
from .base import BaseLayer


class CrossEntropyLoss(BaseLayer):
    """
    Слой-функция потерь, категориальная кросс-энтропия для задачи класификации на
    N классов.
    Применяет softmax к входным данных.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.EPS = 1e-15
        self.pred = None
        self.target = None
        

    def forward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Принимает два тензора - предсказанные логиты классов и правильные классы.
        Prediction и target имеют одинаковый размер вида
         [B, C, 1, ... 1] т.е. имеют 2 оси (batch size и channels) длины больше 1
          и любое количество осей единичной длины.
        В predictions находятся логиты, т.е. к ним должен применяться softmax перед вычислением
         энтропии.
        В target[B, C, (1, ..., 1)] находится 1, если объект B принадлежит классу C, иначе 0 (one-hot представление).
        Возвращает np.array[B] c лоссом по каждому объекту в мини-батче.

        P[B, c] = exp(pred[B, c]) / Sum[c](exp(pred[B, c])
        Loss[B] = - Sum[c]log( prob[B, C] * target[B, C]) ) = -log(prob[B, C_correct])
        """
        
        output = np.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            output[i] = -np.sum(np.log((np.exp(pred[i]) / np.sum(np.exp(pred[i])))[target[i] == 1]))

        self.pred = pred
        self.target = target
        
        return output

    def backward(self) -> np.ndarray:
        """
        Возвращает градиент лосса по pred, т.е. первому аргументу .forward
        Не принимает никакого градиента по определению.
        """
        
        pred = self.pred
        target = self.target

        grad_input = np.zeros(pred.shape)
        for i in range(pred.shape[0]):
            grad_input[i] = np.exp(pred[i]) / np.sum(np.exp(pred[i]))
            grad_input[i][target[i] == 1] -= 1

        return grad_input
