import torch
import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1 = nn.Sequential(
                 nn.Linear(1280, 800),
                 nn.ReLU(),
                 nn.Linear(800, 300),
                 nn.ReLU(),
                 nn.Linear(300, 100),
                 nn.ReLU(),
                 nn.Linear(100, 2),
                 nn.Sigmoid()
        )

    def forward(self, x):
        output = (self.hidden1(x))
        return output

class Net_orig(nn.Module):
    def __init__(self):
        super(Net_orig,self).__init__()
        self.hidden1 = nn.Sequential(
                 nn.Linear(1280*2, 800),
                 nn.ReLU(),
                 nn.Linear(800, 300),
                 nn.ReLU(),
                 nn.Linear(300, 100),
                 nn.ReLU(),
                 nn.Linear(100, 2),
                 nn.Sigmoid()
        )

    def forward(self, x):
        output = (self.hidden1(x))
        return output


if __name__==('__main__'):
   print (Net())
