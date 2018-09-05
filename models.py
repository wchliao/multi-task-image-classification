import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, layers):
        super(_Encoder, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)

        return x


class _Decoder(nn.Module):
    def __init__(self, output_size):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, output_size, kernel_size=1)
        )

    def forward(self, input):
        x = self.layers(input)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.squeeze()

        return x


class _Model(nn.Module):
    def __init__(self, output_size, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(output_size=output_size)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x


def Model(num_classes, num_channels):
    layers = [
        nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ]

    if isinstance(num_classes, list):
        encoders = [_Encoder(layers=layers) for _ in num_classes]
        return [_Model(output_size=cls, encoder=encoder) for cls, encoder in zip(num_classes, encoders)]
    else:
        encoder = _Encoder(layers=layers)
        return _Model(output_size=num_classes, encoder=encoder)
