import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, layers):
        super(_Encoder, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)

        return x


class _Decoder(nn.Module):
    def __init__(self, num_classes):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.AvgPool2d(kernel_size=8),
        )

    def forward(self, input):
        x = self.layers(input)
        x = torch.squeeze(x)

        return x


class _Model(nn.Module):
    def __init__(self, num_classes, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(num_classes=num_classes)


    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x


def Model(num_classes, num_tasks=1):

    layers = [
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

    if num_tasks == 1:
        encoder = _Encoder(layers=layers)
        return _Model(num_classes=num_classes, encoder=encoder)
    else:
        encoders = [_Encoder(layers=layers) for _ in range(num_tasks)]
        return [_Model(num_classes=num_classes, encoder=encoder) for encoder in encoders]
