import numpy as np

import src.models.CNN.data_gen as data_gen
import src.constants as const




import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and a sigmoid.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        h = func.relu(self.fc2(h))
        h = func.relu(self.fc3(h))
        h = torch.sigmoid(self.fc4(h))
        return h

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, optimizer):
        self.train()

        inputs = torch.from_numpy(data.x_train)
        targets = torch.from_numpy(data.y_train)
        targets = targets.reshape((len(targets), 1))
        obj_val = loss(self.forward(inputs), targets)

        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(data.x_test)
            targets = torch.from_numpy(data.y_test)
            targets = targets.reshape((len(targets), 1))
            cross_val = loss(self.forward(inputs), targets)

        return cross_val.item()

    def accuracy(self, x, y):
        self.eval()
        with torch.no_grad():
            # Convert the input data to a tensor and pass into the model
            output = self(torch.from_numpy(x))
            prediction = np.round(output.numpy()).flatten()  # Round to closest values

            differences = np.abs(y - prediction)
            num_wrong =  np.count_nonzero(differences, axis=0)
        return len(x) - num_wrong, len(x)


if __name__ == "__main__":
    x = data_gen.Data(const.TRAIN_DATASET, const.TEST_DATASET)
    Net()