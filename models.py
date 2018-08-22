import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        x = self.convs(input)
        x = x.view(x.size(0), -1)

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(32*24*24, 256),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, input):
        x = self.fcs(input)

        return x


class SingleTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(SingleTaskModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)


    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x
