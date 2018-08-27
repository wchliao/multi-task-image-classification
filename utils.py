import numpy as np
import json
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


class BaseDataLoader:
    def __init__(self, batch_size, train, shuffle):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CIFAR10Loader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True):
        super(CIFAR10Loader, self).__init__(batch_size, train, shuffle)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.task_dataloader = None

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle


    def _create_TaskDataLoaders(self):
        images = []
        labels = []

        for batch_images, batch_labels in self.dataloader:
            for i in batch_images:
                images.append(i)
            for l in batch_labels:
                labels.append(l)

        self.task_dataloader = []
        for t in range(10):
            dataset = CustomDataset(data=images.copy(), labels=[(c == t).long() for c in labels])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.task_dataloader.append(dataloader)


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(10)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(10))
        else:
            assert task in list(range(10)), 'Unknown task: {}'.format(task)
            labels = [0 for _ in range(10)]
            labels[task] = 1
            return labels


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


class CIFAR100Loader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True):
        super(CIFAR100Loader, self).__init__(batch_size, train, shuffle)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )

        dataset = torchvision.datasets.CIFAR100(root='./data', train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.task_dataloader = None
        self.labels = None

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle


    def _create_TaskDataLoaders(self):
        with open('CIFAR100_fine2coarse.json', 'r') as f:
            data_info = json.load(f)

        images = [[] for _ in range(20)]
        labels = [[] for _ in range(20)]

        for batch_images, batch_labels in self.dataloader:
            for i, l in zip(batch_images, batch_labels):
                images[data_info['task'][l]].append(i)
                labels[data_info['task'][l]].append(data_info['subclass'][l])

        self.task_dataloader = []
        for task_images, task_labels in zip(images, labels):
            dataset = CustomDataset(data=task_images, labels=task_labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.task_dataloader.append(dataloader)


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(20)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def _create_labels(self):
        with open('CIFAR100_fine2coarse.json', 'r') as f:
            data_info = json.load(f)

        self.labels = [[] for _ in range(20)]
        for i, t in enumerate(data_info['task']):
            self.labels[t].append(i)


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(100))
        else:
            assert task in list(range(20)), 'Unknown task: {}'.format(task)
            if self.labels is None:
                self._create_labels()
            return self.labels[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = len(self.dataloaders[0])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task