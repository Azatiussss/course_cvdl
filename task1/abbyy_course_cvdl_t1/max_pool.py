from tkinter import constants
import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    @staticmethod
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        padding = np.array([(0, 0) for i in range(len(tensor.shape))])
        padding[axis] = (one_size_pad, one_size_pad)
        return np.pad(tensor, padding, constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        self.input = input
        m, n_C, n_H_prev, n_W_prev = input.shape
        X_pad = self._pad_neg_inf(input, self.padding)

        n_H = (n_H_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        n_W = (n_W_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((m, n_C, n_H, n_W))
        
        for i in range(m):    
            img = X_pad[i]
            for c in range(n_C):
                for w in range(n_W):
                    for h in range(n_H):
                        w_slice = slice(self.stride * w, self.stride * w + self.kernel_size)
                        h_slice = slice(self.stride * h, self.stride * h + self.kernel_size)
                        img_slice = img[c, h_slice, w_slice]
                        output[i, c, h, w] = np.max(img_slice)
                        
        return output


    def backward(self, output_grad: np.ndarray)->np.ndarray:
        input = self.input
        m, n_C, n_H_prev, n_W_prev = input.shape

        X_pad = self._pad_neg_inf(input, self.padding)
    
        n_H = (n_H_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        n_W = (n_W_prev + 2 * self.padding - self.kernel_size) // self.stride + 1
        

        output_grad = output_grad.reshape((m, n_C, n_H, n_W))

        grad_input = np.zeros_like(X_pad)
        
        for i in range(m):
            img = X_pad[i]
            for c in range(n_C):
                for w in range(n_W):
                    for h in range(n_H):
                        w_slice = slice(self.stride * w, self.stride * w + self.kernel_size)
                        h_slice = slice(self.stride * h, self.stride * h + self.kernel_size)
                        max_index = np.unravel_index(np.argmax(img[c, h_slice, w_slice]), img[c, h_slice, w_slice].shape)
                        grad_input[i, c, h_slice, w_slice][max_index[0], max_index[1]] += output_grad[i, c, h, w] #* np.max(img[c, h_slice, w_slice])
        return grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

