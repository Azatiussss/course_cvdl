from torch import nn
import torch
from torch.nn.functional import pad
from abbyy_course_cvdl_t2.head import CenterNetHead
from abbyy_course_cvdl_t2.backbone import ResnetBackbone
from abbyy_course_cvdl_t2.convert import PointsToObjects


class PointsNonMaxSuppression(nn.Module):
    """
    Описан в From points to bounding boxes, фильтрует находящиеся
    рядом объекты.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, points):
        padding = self.kernel_size // 2
        padded_probs = pad(torch.max(points[:, :-4], dim=1)[0], (padding, padding, padding, padding))
        output = torch.zeros_like(points)
        for b in range(points.shape[0]):
            for i in range(padding, points.shape[-2] + padding):
                for j in range(padding, points.shape[-1] + padding):
                    max_index = torch.argmax(padded_probs[b, i:i+self.kernel_size, j:j+self.kernel_size])
                    if max_index // self.kernel_size == max_index % self.kernel_size == self.kernel_size // 2:
                        output[b, :, i - padding, j - padding] = points[b, :, i - padding, j - padding]
                
        return output


class ScaleObjects(nn.Module):
    """
    Объекты имеют размеры в пикселях, и размер входа в сеть
    в несколько раз больше (R, output_stride), чем размер выхода.
    Из-за этого все объекты, полученные с помощью PointsToObjects
    имеют меньший размер.
    Чтобы это компенисровать, надо увеличить размеры объектов.
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, objects):
        b, n, d6 = objects.shape
        objects[:, :, :4] *= self.scale
        return objects


class CenterNet(nn.Module):
    """
    Детектор объектов из статьи 'Objects as Points': https://arxiv.org/pdf/1904.07850.pdf
    """
    def __init__(self, pretrained=True, head_kwargs={}, nms_kwargs={}, points_to_objects_kwargs={}):
        super().__init__()
        self.backbone = ResnetBackbone(pretrained)
        self.head = CenterNetHead(**head_kwargs)
        self.return_objects = torch.nn.Sequential(
            PointsNonMaxSuppression(**nms_kwargs),
            PointsToObjects(**points_to_objects_kwargs),
            ScaleObjects()
        )

    def forward(self, input_t, return_objects=False):
        x = input_t
        x = self.backbone(x)
        x = self.head(x)
        if return_objects:
            x = self.return_objects(x)
        return x
