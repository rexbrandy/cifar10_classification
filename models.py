import torch
from torch import nn
from torch.nn import functional as f

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Feed forward
        '''
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)
        '''

        self.fc1 = nn.Linear(16 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 240)
        self.fc3 = nn.Linear(240, 60)
        self.fc4 = nn.Linear(60, 10)


    def forward(self, x):
        output = self.pool(f.relu(self.conv1(x)))
        output = self.pool(f.relu(self.conv2(output)))

        output = torch.flatten(output, 1) # flatten all dimensions except batch size

        output = f.relu(self.fc1(output))
        output = f.relu(self.fc2(output))
        output = f.relu(self.fc3(output))
        output = self.fc4(output)
        return output

