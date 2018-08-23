import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, input):
        x = self.convs(input)
        x = x.view(x.size(0), -1)

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(64*5*5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, input):
        x = self.fcs(input)

        return x


class StandardModel(nn.Module):
    def __init__(self, num_classes):
        super(StandardModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)


    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x


class SharedEncoderModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SharedEncoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder(num_classes=num_classes)


    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x
