import torch
import torchvision


def DataLoader(batch_size=128, train=True, shuffle=True, num_workers=2):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                           download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)

    return dataloader