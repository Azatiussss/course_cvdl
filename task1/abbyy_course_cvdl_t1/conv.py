import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.parameters.append(np.ones((out_channels, in_channels, kernel_size, kernel_size)))
        self.parameters.append(np.zeros(out_channels))

        self.parameters_grads.append(np.zeros_like(self.parameters[0]))
        self.parameters_grads.append(np.zeros_like(self.parameters[1]))
        


    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        padding = np.array([(0, 0) for i in range(len(tensor.shape))])
        padding[axis] = (one_side_pad, one_side_pad)
        return np.pad(tensor, padding)

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1

        return np.sum(np.multiply(input, kernel))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        m, _, n_H, n_W = input.shape
        padding_size = (self.kernel_size - 1) // 2 
        X_pad = self._pad_zeros(input, padding_size)
        
        output = np.zeros((m, self.out_channels, n_H, n_W))
        
        for i in range(m):    
            img = X_pad[i]
            for c in range(self.out_channels):
                fil = self.parameters[0][c]
                b = self.parameters[1][c]
                for w in range(n_W):
                    for h in range(n_H):
                        w_slice = slice(w, w + self.kernel_size)
                        h_slice = slice(h, h + self.kernel_size)
                        img_slice = img[:, h_slice, w_slice]
                        output[i, c, h, w] += self._cross_correlate(img_slice, fil) + b
                        
        return output

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        input = self.input
        m, _, n_H, n_W = input.shape
        
        padding_size = (self.kernel_size - 1) // 2
        grad_input = np.zeros((m, self.in_channels, n_H + 2 * padding_size, n_W + 2 * padding_size))
        X_pad = self._pad_zeros(input, padding_size)
        
        for i in range(m):     
            img = X_pad[i]
            for c in range(self.out_channels):
                for w in range(n_W):
                    for h in range(n_H):
                        w_slice = slice(w, w + self.kernel_size)
                        h_slice = slice(h, h + self.kernel_size)
                        img_slice = img[:, h_slice, w_slice]
                        grad_input[i, :, h_slice, w_slice] += self.parameters[0][c] * output_grad[i, c, h, w]
                        self.parameters_grads[0][c, :, :, :] += img_slice * output_grad[i, c, h, w]
                        self.parameters_grads[1][c] += output_grad[i, c, h, w]

        return grad_input[:, :, padding_size:-padding_size, padding_size:-padding_size]
