---
layout: default
title: "09 Well-Organized DL Code "
parent: deep-learning
permalink: /dl/sa/standalone-09/
subtitle: Dl and MLP 
use_math : true
---

# [Standalone DLç] #15 Lab - 09 - How to Write Well-Organized DL Code from Scratch 

## Data Preparation

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np

# transform : 이미지를 tensor로 변환해주는 작업 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                        shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

### Model Architecture

```python
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act

        self.fc = nn.Linear(self.in_dim, self.hid_dim)
        self.linears = nn.ModuleList()

        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim)) # hid_dim으로 들어와서 hid_dim으로 나간다 
        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

        if self.act == 'relu': 
            self.act = nn.ReLU() # 원하는 activation function이 있으면 condition을 더 추가할 수 있음 

    def forward(self, x):
        x = self.act(self.fc(x))
        for fc in self.linears:
            x = self.act(fc(x))
        x = self.fc2(x)
        return x

net = MLP(3072, 10, 100, 4, 'relu')
print(net) # 모델 구조가 출력된다 

# 아래와 같은 아웃풋이 출력됨 
MLP(
  (fc): Linear(in_features=3072, out_features=100, bias=True)
  (linears): ModuleList(
    (0-2): 3 x Linear(in_features=100, out_features=100, bias=True)
  )
  (fc2): Linear(in_features=100, out_features=10, bias=True)
  (act): ReLU()
)
```

```python
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

```python
for epoch in range(2):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data 
        #print(inputs.shape) # torch.Size([4, 3, 32, 32]) 
        # 4x3x32x32=12288 개의 tensor로 이루어져 있는 상태 

        inputs = inputs.view(-1, 3072) # -1 자리에는 12288/3072=4 값이 들어가게 됨 
        #print(inputs.shape) # torch.Size([4, 3072])

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

```

출력은 다음과 같다 
`[1,  2000] loss: 1.755
[1,  4000] loss: 1.663
[1,  6000] loss: 1.617
[1,  8000] loss: 1.593`
-> 모델을 학습할 수 있는 상태가 됐다 (loss도 줄어드는 것 같고 .. ) 

```python
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(-1, 3072) # 추가해줌 
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

출력 
`Accuracy of the network on the 10000 test images: 47 %`

```python
correct = 0
total = 0

with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.view(-1, 3072)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

출력
`Accuracy of the network on the 10000 test images: 45 %`

### Experiment

```python
def experiment(args):

    net = MLP(args.in_dim, args.out_dim, args.hid_dim, args.n_layer, args.act)
    net.cuda()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mm)

    for epoch in range(args.epoch):  

        # ==== Train ===== #
        net.train()

        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad() 

            # get the inputs
            inputs, labels = data
            inputs = inputs.view(-1, 3072)

            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # ==== Validation ====== #
        net.eval()

        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.view(-1, 3072)

                images = images.cuda()
                labels = labels.cuda()

                outputs = net(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(valloader)
            val_acc = 100 * correct / total

        print('Epoch {}, Train Loss: {}, Val Loss: {}, Val Acc: {}'.format(epoch, train_loss, val_loss, val_acc ))

    # ===== Evaluation ===== #
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 3072)
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

    return train_loss, val_loss, val_acc, test_acc
```

```python
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.n_layer = 5
args.in_dim = 3072
args.out_dim = 10
args.hid_dim = 100
args.act = 'relu'

args.lr = 0.001
args.mm = 0.9
args.epoch = 2

list_var1 = [4, 5, 6]
list_var2 = [50, 100, 150]

for var1 in list_var1:
    for var2 in list_var2:
        args.n_layer = var1
        args.hid_dim = var2
        result = experiment(args)
        print(result) 
```

출력값 

MLP(
(fc): Linear(in_features=3072, out_features=50, bias=True)
(linears): ModuleList(
(0-2): 3 x Linear(in_features=50, out_features=50, bias=True)
)
(fc2): Linear(in_features=50, out_features=10, bias=True)
(act): ReLU()
)
[1,  2000] loss: 2.202
[1,  4000] loss: 1.900
[1,  6000] loss: 1.791
[1,  8000] loss: 1.722
[1, 10000] loss: 1.693
Epoch 0, Train Loss: 18616.791673600674, Val Loss: 1.7005987400531768, Val Acc: 39.32
[2,  2000] loss: 1.626
[2,  4000] loss: 1.608
[2,  6000] loss: 1.584
[2,  8000] loss: 1.584
[2, 10000] loss: 1.544
Epoch 1, Train Loss: 15891.401199430227, Val Loss: 1.564451172888279, Val Acc: 43.8
(15891.401199430227, 1.564451172888279, 43.8, 45.1)
MLP(
(fc): Linear(in_features=3072, out_features=100, bias=True)
(linears): ModuleList(
(0-2): 3 x Linear(in_features=100, out_features=100, bias=True)
)
(fc2): Linear(in_features=100, out_features=10, bias=True)
(act): ReLU()
)
[1,  2000] loss: 2.198
[1,  4000] loss: 1.924
[1,  6000] loss: 1.792
[1,  8000] loss: 1.727
[1, 10000] loss: 1.644
Epoch 0, Train Loss: 18569.359001129866, Val Loss: 1.6861703926563263, Val Acc: 38.98
[2,  2000] loss: 1.594
[2,  4000] loss: 1.579
[2,  6000] loss: 1.561
[2,  8000] loss: 1.545
[2, 10000] loss: 1.525
Epoch 1, Train Loss: 15606.880003154278, Val Loss: 1.5393070907831192, Val Acc: 44.97
(15606.880003154278, 1.5393070907831192, 44.97, 45.94)
MLP(
(fc): Linear(in_features=3072, out_features=150, bias=True)
(linears): ModuleList(
(0-2): 3 x Linear(in_features=150, out_features=150, bias=True)
)
(fc2): Linear(in_features=150, out_features=10, bias=True)
(act): ReLU()
)
[1,  2000] loss: 2.179
[1,  4000] loss: 1.895
[1,  6000] loss: 1.760
[1,  8000] loss: 1.684
[1, 10000] loss: 1.646
Epoch 0, Train Loss: 18327.9337708354, Val Loss: 1.644619299721718, Val Acc: 40.36
[2,  2000] loss: 1.570
[2,  4000] loss: 1.566
[2,  6000] loss: 1.520
[2,  8000] loss: 1.507
[2, 10000] loss: 1.500
Epoch 1, Train Loss: 15323.40272564441, Val Loss: 1.521736914730072, Val Acc: 46.1

여기서 파라미터를 바꿔가며 성능을 향상시킬 수 있다.