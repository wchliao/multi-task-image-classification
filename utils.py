import torch
import torchvision


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class CIFAR10Loader:
    def __init__(self, batch_size=128, train=True, shuffle=True):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.taskloader = None

        if train:
            self._len = 50000
        else:
            self._len = 10000

        self.batch_size = batch_size
        self.shuffle = shuffle


    def _create_taskloaders(self):
        images = []
        labels = []

        for batch_images, batch_labels in self.dataloader:
            for i in batch_images:
                images.append(i)
            for l in batch_labels:
                labels.append(l)

        self.taskloader = []
        for t in range(10):
            dataset = CustomDataset(data=images.copy(), labels=[(c == t).long() for c in labels])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.taskloader.append(dataloader)


    def get_loader(self, task=None):
        if task is None:
            return self.dataloader
        else:
            assert task in list(range(10)), 'Unknown task: {}'.format(task)
            if self.taskloader is None:
                self._create_taskloaders()
            return self.taskloader[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


def DataLoader(batch_size=128, CIFAR10=True, train=True, shuffle=True):
    if CIFAR10:
        return CIFAR10Loader(batch_size=batch_size, train=train, shuffle=shuffle)
    else:
        raise NotImplementedError