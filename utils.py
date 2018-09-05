import numpy as np
import os
import cv2
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

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
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


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes_single(self):
        return 10


    @property
    def num_classes_multi(self):
        return [2 for _ in range(10)]


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


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes_single(self):
        return 100


    @property
    def num_classes_multi(self):
        return [5 for _ in range(20)]


class OmniglotLoader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True):
        super(OmniglotLoader, self).__init__(batch_size, train, shuffle)
        omniglot_path = './data/omniglot'

        if os.path.isdir(omniglot_path):
            print('Files already downloaded and verified')
        else:
            raise FileNotFoundError('Omniglot dataset not found. Please download it and put it under \'{}\''.format(omniglot_path))

        images = []
        labels = []
        self._len = 0
        self.task_dataloader = []
        self.num_classes = []

        for p in [os.path.join(omniglot_path, 'images_background'), os.path.join(omniglot_path, 'images_evaluation')]:
            for task_path in sorted(os.listdir(p)):
                task_path = os.path.join(p, task_path)
                task_images = []
                task_labels = []
                for i, cls_path in enumerate(sorted(os.listdir(task_path))):
                    cls_path = os.path.join(task_path, cls_path)
                    ims = [cv2.imread(os.path.join(cls_path, filename), cv2.IMREAD_GRAYSCALE) / 255 for filename in sorted(os.listdir(cls_path))]

                    if train:
                        ims = ims[:int(len(ims)*0.8)]
                    else:
                        ims = ims[int(len(ims)*0.8):]

                    self._len += len(ims)
                    task_images += ims
                    task_labels += [i for _ in range(len(ims))]

                task_images = np.expand_dims(task_images, 1)
                dataset = CustomDataset(data=torch.Tensor(task_images).float(), labels=torch.Tensor(task_labels).long())
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
                self.task_dataloader.append(dataloader)

                self.num_classes.append(len(np.unique(task_labels)))

                images.append(task_images)
                labels.append(task_labels)

        images = np.concatenate(images)
        labels = np.concatenate(labels)

        new_label = 0
        new_labels = [new_label]
        for prev_label, label in zip(labels[:-1], labels[1:]):
            if prev_label != label:
                new_label += 1
            new_labels.append(new_label)

        new_labels = torch.Tensor(new_labels).long()
        images = torch.from_numpy(images).float()

        dataset = CustomDataset(data=images, labels=new_labels)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        self.labels = []
        cnter = 0
        for num_classes in self.num_classes:
            self.labels.append(list(range(cnter, cnter + num_classes)))
            cnter += num_classes

        self.batch_size = batch_size
        self.shuffle = shuffle


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(50)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(50))
        else:
            assert task in list(range(50)), 'Unknown task: {}'.format(task)
            return self.labels[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 1


    @property
    def num_classes_single(self):
        return sum(self.num_classes)


    @property
    def num_classes_multi(self):
        return self.num_classes


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