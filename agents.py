import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from models import SingleTaskModel


class BaseAgent:
    def __init__(self):
        pass

    def train(self, train_data, test_data, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass


class StandardAgent(BaseAgent):
    def __init__(self):
        super(StandardAgent, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SingleTaskModel(num_classes=10).to(self.device)


    def train(self, train_data, test_data, save_history=False, save_path='.', verbose=False):
        criterion =nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        accuracy = []

        for epoch in range(30):
            for _, (inputs, labels) in enumerate(train_data):
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
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, 'history.json')

            with open(filename, 'w') as f:
                json.dump(accuracy, f)


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
