# define the multi-task neural network model

import torch.nn as nn
import torch

class Network_XRD(nn.Module):
    def __init__(self):
        super(Network_XRD, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=11,stride=2,padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=11,stride=2,padding=0),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2, padding=0),
            nn.Flatten()
        )
        
        
        nf = 3584
        d1 = 0.2
        d2 = 0.2
        d3 = 0.2
        d4 = 0.2
        self.fc1 = nn.Sequential(
            nn.Linear(nf, 1024),
            nn.Dropout(d1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(d2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(d3),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Dropout(d4),
            nn.Softmax(dim=1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(nf, 1024),
            nn.Dropout(d1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(d2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(d3),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Dropout(d4),
            nn.Softmax(dim=1)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(nf, 1024),
            nn.Dropout(d1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(d2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(d3),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Dropout(d4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        ff = self.layers(x)

        f_1 = self.fc1(ff)
        f_2 = self.fc2(ff)
        f_3 = self.fc3(ff)
        return f_1,f_2,f_3

