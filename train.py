import torch
import modelA
import modelB

mA = modelA.modelA().to('cuda:0')
mB = modelB.modelB().to('cuda:1')

optimizer = torch.optim.SGD(list(mA.parameters()) + list(mB.parameters()), 0.1,
                                momentum=0.05,
                                weight_decay=0.05)
optimizer.zero_grad()

data = torch.rand(1, 1000).to('cuda:0')

x = mA(data)
x = mB(x.to('cuda:1'))

target = torch.rand(1,2000).to('cuda:1')

lossf = torch.nn.L1Loss()

loss = lossf(x, target)
optimizer.step()
