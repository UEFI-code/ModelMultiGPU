import torch

class modelA(torch.nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.li1 = torch.nn.Linear(1000, 2000)
    def forward(self, x):
        return self.li1(x)

