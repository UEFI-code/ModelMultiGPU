import torch

class modelB(torch.nn.Module):
    def __init__(self):
        super(modelB, self).__init__()
        self.li1 = torch.nn.Linear(2000, 2000)
    def forward(self, x):
        return self.li1(x)
