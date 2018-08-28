import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, shared_layers):
        super(_Encoder, self).__init__()

        layers = []
        for layer in shared_layers:
            layers.append(layer)
            layers.append(nn.BatchNorm2d(layer.out_channels))
            layers.append(nn.ReLU())

        self.convs = nn.Sequential(*layers)

    def forward(self, input):
        x = self.convs(input)
        x = x.view(x.size(0), -1)

        return x


class _Decoder(nn.Module):
    def __init__(self, num_classes):
        super(_Decoder, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(64*5*5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, input):
        x = self.fcs(input)

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

    convs = [
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3, stride=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.Conv2d(64, 64, kernel_size=3, stride=2)
    ]

    if num_tasks == 1:
        encoder = _Encoder(shared_layers=convs)
        return _Model(num_classes=num_classes, encoder=encoder)
    else:
        encoders = [_Encoder(shared_layers=convs) for _ in range(num_tasks)]
        return [_Model(num_classes=num_classes, encoder=encoder) for encoder in encoders]
