import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from models import Encoder
from models import StandardModel, SharedEncoderModel


class BaseAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, test_data, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass


class SingleTaskAgent(BaseAgent):
    def __init__(self, num_classes):
        super(SingleTaskAgent, self).__init__()
        self.model = StandardModel(num_classes=num_classes).to(self.device)


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, data):
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            return correct / total


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'model')

        torch.save(self.model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            filename = os.path.join(save_path, 'model')
            self.model.load_state_dict(torch.load(filename))


class StandardAgent(SingleTaskAgent):
    def __init__(self, CIFAR10):
        if CIFAR10:
            super(StandardAgent, self).__init__(num_classes=10)
            self.eval = self._eval_CIFAR10
        else:
            super(StandardAgent, self).__init__(num_classes=100)
            self.eval = self._eval_CIFAR100


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def _eval_CIFAR10(self, data):
        correct = [0 for _ in range(10)]
        total = 0

        with torch.no_grad():
            for inputs, labels in data.get_loader():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)

                for c in range(10):
                    correct[c] += ((predict_labels == c) == (labels == c)).sum().item()

            return [c / total for c in correct]


    def _eval_CIFAR100(self, data):
        correct = [0 for _ in range(20)]
        total = [0 for _ in range(20)]

        with torch.no_grad():
            for t in range(20):
                task_labels = data.get_labels(t)
                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predict_labels = torch.max(outputs[:, task_labels].detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

            return [c / t for c, t in zip(correct, total)]


class MultiTaskSeparateAgent:
    def __init__(self, num_tasks, num_classes, task_prob=None):
        super(MultiTaskSeparateAgent, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = Encoder()
        self.models = [SharedEncoderModel(encoder=encoder, num_classes=num_classes).to(self.device)
                       for _ in range(num_tasks)]
        self.num_tasks = num_tasks
        self.task_prob = task_prob


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        if self.task_prob is None:
            dataloader = train_data.get_loader('multi-task')
        else:
            dataloader = train_data.get_loader('multi-task', prob=self.task_prob)

        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.Adam(model.parameters()) for model in self.models]
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels, task in dataloader:
                model = self.models[task]
                optimizer = optimizers[task]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def eval(self, data):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        with torch.no_grad():
            for t, model in enumerate(self.models):
                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predict_labels = torch.max(outputs.detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

            return [c / t for c, t in zip(correct, total)]


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for t, model in enumerate(self.models):
            filename = os.path.join(save_path, 'model{}'.format(t))
            torch.save(model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            for t, model in enumerate(self.models):
                filename = os.path.join(save_path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))


class MultiTaskJointAgent(MultiTaskSeparateAgent):
    """
    MultiTaskJointAgent can only be used in tasks that share the same inputs.
    For CIFAR datasets, it can only apply to CIFAR-10 multi-task experiments.
    CIFAR-100 multi-task experiments are not applicable.
    """

    def __init__(self, num_tasks, num_classes, loss_weight=None):
        super(MultiTaskJointAgent, self).__init__(num_tasks, num_classes)

        if loss_weight is None:
            self.loss_weight = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        else:
            self.loss_weight = torch.Tensor(loss_weight).to(self.device)


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        dataloader = train_data.get_loader()
        criterion = nn.CrossEntropyLoss()

        parameters = []
        for model in self.models:
            parameters += model.parameters()
        parameters = set(parameters)
        optimizer = optim.Adam(parameters)

        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss = 0.

                for t, model in enumerate(self.models):
                    outputs = model(inputs)
                    loss += self.loss_weight[t] * criterion(outputs, (labels == t).long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)
