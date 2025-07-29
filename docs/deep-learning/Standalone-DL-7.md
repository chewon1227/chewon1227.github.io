---
layout: default
title: "07 GPU with Pytorch "
parent: deep-learning
permalink: /dl/sa/standalone-07/
subtitle: Dl and MLP 
use_math : true
---

# [Standalone DLç] #12 Lab - GPU with Pytorch

## Why GPU is fast?

- (NVIDIA) GPU : cuda 라는 작지만 여러 개의 processor 들로 구성이 되어 있음
- why Fast ? = CPU는 iterative하게 for 문을 돌지만, GPUsms 연산을 병렬적으로 처리할 수 있음 (독립적 연산 진행)
    - CPU : 박사생 4명이서 일하는 것. GPU : 학부생 4000명이서 일하는 것.

<br />

## How to use GPU with Pytorch

```bash
model = MLPModel(784, 10, [1000])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu)
model.to(device) # done ! 

input_x = input_x.to(device)  
true_y = true_y.to(device) # 데이터도 gpu로 보내는 것 
```

## Code - MNIST

### Data Preparation

```python
import torch
from torchvision import datasets, transforms
import torch.nn as nn

# 데이터를 특정 수식에 맞추어 준비한

batch_size = 128
train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_dataset =  datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
```

### Model Architecture & Cost Function

```python
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=784, out_features=10, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x

cls_loss = nn.CrossEntropyLoss()
```

### Train by CPU

```python
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time 

# ====== Construct Model ====== #
model = LinearModel()
print('Number of {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ===== Construct Optimizer ====== #
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)

list_epoch = []
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 30
for i in range(epoch):
    ts = time.time()

    # ====== Train ====== #
    train_loss = 0
    model.train()

    for input_X, true_y in train_loader:
        optimizer.zero_grad()

        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        pred_y = model(input_X)

        loss = cls_loss(pred_y.squeeze(), true_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().numpy()
    train_loss = train_loss / len(train_loader)
    list_train_loss.append(train_loss)
    list_epoch.append(i)

    # ====== Validation ====== #
    val_loss = 0
    model.eval()
    # optimizer.zero_grad() [

    with torch.no_grad():
        for input_X, true_y in val_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            pred_y = model(input_X)

            loss = cls_loss(pred_y.squeeze(), true_y)
            val_loss += loss.detach().numpy()
        val_loss = val_loss / len(val_loader)
        list_val_loss.append(val_loss)

    # ====== Evaluation ======= #
    correct = 0
    model.eval()
    # optimizer.zero_grad() 

    with torch.no_grad():
        for input_X, true_y in test_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
            correct += pred_y.eq(true_y).sum()

        acc = correct.numpy() / len(test_loader.dataset)
        list_acc.append(acc)
        list_acc_epoch.append(i)

        te = time.time()
        print(te-ts)

    print('Epoch: {}, Train Loss: {}, Val Loss: {}, Test Acc: {}%, {:3.1f}'.format(i, train_loss, val_loss, acc*100, te-ts))
```

CPU를 사용할 때의 epoch 당 걸리는 시간 

```bash
Number of 7850 parameters
Epoch: 0, Train Loss: 0.7371658086776733, Val Loss: 0.4755309522151947, Test Acc: 88.38000000000001%, 14.4
Epoch: 1, Train Loss: 0.442359983921051, Val Loss: 0.4023783504962921, Test Acc: 89.88000000000001%, 14.7
Epoch: 2, Train Loss: 0.39372873306274414, Val Loss: 0.3729289174079895, Test Acc: 90.16%, 15.2
Epoch: 3, Train Loss: 0.369748592376709, Val Loss: 0.35486114025115967, Test Acc: 90.52%, 14.6
Epoch: 4, Train Loss: 0.35420820116996765, Val Loss: 0.3435230851173401, Test Acc: 90.74%, 14.4
```

### Train by GPU

우선 device 정의부터 

```python
print(torch.cuda.is_available()) # pytorch가 gpu를 잘 인식하고 있는가? (하드웨어 가속기를 cpu로 설정하면 False가 나올 것)
```

```python
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time 

# ====== Construct Model ====== #
model = LinearModel()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # device를 정의한다
model.to(device) # gpu에 주기 위해 device에 보낸다 
print('Number of {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ===== Construct Optimizer ====== #
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)

list_epoch = []
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 30
for i in range(epoch):
    ts = time.time()

    # ====== Train ====== #
    train_loss = 0
    model.train()

    for input_X, true_y in train_loader:
        optimizer.zero_grad() 

        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        input_X = input_X.to(device) # device에 들어가도록 
        pred_y = model(input_X)
        true_y = true_y.to(device)

        loss = cls_loss(pred_y.squeeze(), true_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() # gpu에서 계산한 것을 numpy 취하면 에러. detach 대신 item을 사용하면 gpu, cpu 모두 커버 가능 (detach().numpy() -> item())
    train_loss = train_loss / len(train_loader)
    list_train_loss.append(train_loss)
    list_epoch.append(i)

    # ====== Validation ====== #
    val_loss = 0
    model.eval()

    with torch.no_grad(): 
        for input_X, true_y in val_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            input_X = input_X.to(device)
            pred_y = model(input_X)
            true_y = true_y.to(device)

            loss = cls_loss(pred_y.squeeze(), true_y)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        list_val_loss.append(val_loss)

    # ====== Evaluation ======= #
    correct = 0
    model.eval()

    with torch.no_grad(): 
        for input_X, true_y in test_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            input_X = input_X.to(device)
            true_y = true_y.to(device)
            pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
            correct += pred_y.eq(true_y).sum()

        acc = correct.item() / len(test_loader.dataset) # 여기도 item()
        list_acc.append(acc)
        list_acc_epoch.append(i)

        te = time.time()

    print('Epoch: {}, Train Loss: {}, Val Loss: {}, Test Acc: {}%, {:3.1f}'.format(i, train_loss, val_loss, acc*100, te-ts))
```

```bash
Number of 7850 parameters
Epoch: 0, Train Loss: 0.7453517858177194, Val Loss: 0.47398917659928524, Test Acc: 88.12%, 14.5
Epoch: 1, Train Loss: 0.439791066338644, Val Loss: 0.40070518813555756, Test Acc: 89.39%, 15.2
Epoch: 2, Train Loss: 0.3917852801358913, Val Loss: 0.37101721725886383, Test Acc: 90.13%, 15.8
Epoch: 3, Train Loss: 0.3679074141604211, Val Loss: 0.3532045691073695, Test Acc: 90.55%, 14.5
Epoch: 4, Train Loss: 0.35277878037651483, Val Loss: 0.3427322418252124, Test Acc: 90.78%, 14.1
```

왜 그닥 차이가 나지 않는가? 

- 우리 데이터셋이 너무 작아서, cpu로 돌리나 gpu로 돌리나 큰 차이가 없을 수 있음
- 통상적으로 gpu가 제대로 작동한다면 5~20배 가량 빨라짐