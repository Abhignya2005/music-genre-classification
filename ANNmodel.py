import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__int__()
        self.fc1 = nn.linear(10, 32)
        self.fc2 = nn.linear(32, 16)
        self.fc3 = nn.linear(16, 3)
    
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc1(x))
        x = self.fc3(x)
        return x 
np=random.seeds(42)
torch.manual_seed(42)

X=np.random_seed