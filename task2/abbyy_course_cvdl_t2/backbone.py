"""
Здесь находится backbone на основе resnet-18, в статье "Objects as Points" он описан в
5.Implementation details/Resnet и в Figure 6-b.
"""
from turtle import forward
from torch import nn
from torchvision.models import resnet18


class HeadlessPretrainedResnet18Encoder(nn.Module):
    """
    Предобученная на imagenet версия resnet, у которой
    нет avg-pool и fc слоев.
    Принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        super().__init__()
        md = resnet18(pretrained=True)
        # все, кроме avgpool и fc
        self.md = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )

    def forward(self, x):
        return self.md(x)


class HeadlessResnet18Encoder(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
        super().__init__()

        class identity_sckipconn(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            
            def forward(self, x):
                return x + self.net(x)

        class up_sckipconn(nn.Module):
            def __init__(self, net, in_channels, out_channels):
                super().__init__()
                self.net = net
                self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
            
            def forward(self, x):
                return self.conv(x) + self.net(x)
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Sequential(identity_sckipconn(nn.Sequential(
                                               nn.Conv2d(64, 64, 3, padding='same'),
                                               nn.Conv2d(64, 64, 3, padding='same')
                                               )), 
                          identity_sckipconn(nn.Sequential(
                                               nn.Conv2d(64, 64, 3, padding='same'),
                                               nn.Conv2d(64, 64, 3, padding='same')
                                               ))),
            *[nn.Sequential(up_sckipconn(nn.Sequential(
                                               nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                                               nn.Conv2d(out_channels, out_channels, 3, padding='same')
                                               ), in_channels, out_channels), 
                          identity_sckipconn(nn.Sequential(
                                               nn.Conv2d(out_channels, out_channels, 3, padding='same'),
                                               nn.Conv2d(out_channels, out_channels, 3, padding='same')
                                               ))) for in_channels, out_channels in zip([64, 128, 256], [128, 256, 512])]

        )

    def forward(self, x):
        return self.net(x)


class UpscaleTwiceLayer(nn.Module):
    """
    Слой, повышающий height и width в 2 раза.
    В реализации из "Objects as Points" используются Transposed Convolutions с
    отсылкой по деталям к https://arxiv.org/pdf/1804.06208.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0)

    def forward(self, x):
        return self.conv(x)


class ResnetBackbone(nn.Module):
    """
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, C, H/R, W/R], где R = 4.
    C может быть выбрано разным, в конструкторе ниже C = 64.
    """
    def __init__(self, pretrained: bool = True, out_channels=64):
        super().__init__()
        # downscale - fully-convolutional сеть, снижающая размерность в 32 раза
        if pretrained:
            self.downscale = HeadlessPretrainedResnet18Encoder()
        else:
            self.downscale = HeadlessResnet18Encoder()

        # upscale - fully-convolutional сеть из UpscaleTwiceLayer слоев, повышающая размерность в 2^3 раз
        downscale_channels = 512 # выход resnet
        channels = [downscale_channels, 256, 128, out_channels]
        layers_up = [
            UpscaleTwiceLayer(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ]
        self.upscale = nn.Sequential(*layers_up)

    def forward(self, x):
        x = self.downscale(x)
        x = self.upscale(x)
        return x

