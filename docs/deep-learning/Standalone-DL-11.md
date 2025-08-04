---
layout: default
title: "11 Advanced Optimizer"
parent: deep-learning
permalink: /dl/sa/standalone-11/
use_math : true
---

# [Standalone DL] 11 Lab - #18 Handling Visualization of Many Experiments 

## Save

### What to Save?

무엇을 시각화할 지 생각해보기 전에, 무엇을 ‘저장’해야 하는지에 대해 생각해보자. 

- args에 있는 세팅 값들을 저장해야 한다.
- epoch에 따른 train loss, val loss, train acc, val acc를 저장해야 한다.
- 최종 train acc, val acc, test acc도 저장해야 한다.

### How to Save?

각 실험 결과들을 dictionary에 넣고 append해나가자 

→ 인터넷이 끊기거나 튕길 시 모든 저장 결과가 날아간다. 

→ JSON 포맷을 활용하자 (모든 언어에서 활용 가능하며, 한 줄 한 줄 저장하면 인터넷이 끊겨도 저장 결과들이 날아가지 않는다) 

```python
import json 
a = {'value1': 5, 'value2':10, 'seq'=[1,2,3,4,5]}

filename = 'test.json'
with open(filename, 'w') as f:
		json.dump(a, f)

with open(filename, 'r') as f:
		result = json.load(f)
		print(result)
```

실험 돌아갈 때마다 각각에 대해 json 파일을 만들자. 구분하기 위해 어떤 방식을 사용하면 좋을까? 

→ 만약 시간이나 랜덤 숫자로 진행한다면, 같은 시험 세팅으로 다시 돌릴 시 다른 Json 파일이 또 생겨서 뭐가 맞는 것인지 알 수 없음

→ 변수 값들을 파일 제목에 같이 넣어주자. 

```python
import hashlib

a = 'my name is rachel'
hash_key = hashlib.shal(a.encode()).hexdigest()[:6] # select 6 string from front 
print(hash_key) # random strings 
```

```python
setting = {'value1': 5, 'value2':10, 'seq'=[1,2,3,4,5]} # 이것을 우리의 변수들이라고 생각하자 
hash_key = hashlib.shal(str(setting).encode()).hexdigest()[:6]
print(hash_key) # 9e23e4 (value에 대한 hash_key가 생성된 것) 

setting = {'value1': 6, 'value2':10, 'seq'=[1,2,3,4,5]}  
hash_key = hashlib.shal(str(setting).encode()).hexdigest()[:6]
print(hash_key) # 6f3bbe (다른 value에 대한 다른 hash_key가 생성된 것) 
```

같은 세팅 값이면 같은 hash 값 , 다른 세팅 값이면 다른 hash 값을 얻을 수 있게 되었다. 

```python
setting = {'value1': 5, 'value2':10, 'seq'=[1,2,3,4,5], 'exp_name': 'exp1'}
exp_name = setting['exp_name']
hash_key = hashlib.shal(str(setting).encode()).hexdigest()[:6]
filename = '{}-{}.json'.format(exp_name, hash_key)
print(filename) # exp1-628721.json 와 같은 식으로, hask_key가 들어간 filename을 얻을 수 있음 
```

## Code

### basic

```python
!mkdir results # results라는 폴더 만들기 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
from copy import deepcopy # Add Deepcopy for args
```

### data preparation

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
partition = {'train': trainset, 'val':valset, 'test':testset}
```

### model architecture

```python
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act, dropout, use_bn, use_xavier):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_xavier = use_xavier

        # ====== Create Linear Layers ====== #
        self.fc1 = nn.Linear(self.in_dim, self.hid_dim)

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.hid_dim))

        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

        # ====== Create Activation Function ====== #
        if self.act == 'relu':
            self.act = nn.ReLU()
        elif self.act == 'tanh':
            self.act == nn.Tanh()
        elif self.act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('no valid activation function selected!')

        # ====== Create Regularization Layer ======= #
        self.dropout = nn.Dropout(self.dropout)
        if self.use_xavier:
            self.xavier_init()

    def forward(self, x):
        x = self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x = self.act(self.linears[i](x))
            x = self.bns[i](x)
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def xavier_init(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0.01)

net = MLP(3072, 10, 100, 4, 'relu', 0.1, True, True) # Testing Model Construction
```

### train, validate, test, exp

```python
def train(net, partition, optimizer, criterion, args):
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                              batch_size=args.train_batch_size,
                                              shuffle=True, num_workers=2)
    net.train()

    correct = 0
    total = 0
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad() 

        # get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 3072)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(trainloader)
    train_acc = 100 * correct / total
    return net, train_loss, train_acc
```

```python
def validate(net, partition, criterion, args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=2)
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
    return val_loss, val_acc
    
def test(net, partition, args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                             batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2)
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
    return test_acc

def experiment(partition, args):

    net = MLP(args.in_dim, args.out_dim, args.hid_dim, args.n_layer, args.act, args.dropout, args.use_bn, args.use_xavier)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        net, train_loss, train_acc = train(net, partition, optimizer, criterion, args)
        val_loss, val_acc = validate(net, partition, criterion, args)
        te = time.time()

        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #

        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch, train_acc, val_acc, train_loss, val_loss, te-ts))

    test_acc = test(net, partition, args)

    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc
    return vars(args), result
    # ===================================== #
```

### Exp results

```python
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd

def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch'] # epoch을 지운다 - epoch이 바뀌어도 파일이 또 생성되지 않고 업데이트 되도록. 
    del setting['test_batch_size'] # 마찬가지 

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = './results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting) # result라는 딕셔너리에 setting도 저장을 해준다 
    with open(filename, 'w') as f:
        json.dump(result, f)

def load_exp_result(exp_name):
    dir_path = './results'
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result) # .drop(columns=[])
    return df

```

### Exp

```python
# ====== Random Seed Initialization ====== #
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_n_layer_hid_dim" # layer과 hid_dim을 바꾼 실험이었다는 뜻으로 작성 

# ====== Model Capacity ====== #
args.in_dim = 3072
args.out_dim = 10
args.hid_dim = 100
args.act = 'relu'

# ====== Regularization ======= #
args.dropout = 0.2
args.use_bn = True
args.l2 = 0.00001
args.use_xavier = True

# ====== Optimizer & Training ====== #
args.optim = 'RMSprop' #'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0015
args.epoch = 10

args.train_batch_size = 256
args.test_batch_size = 1024

# ====== Experiment Variable ====== #
name_var1 = 'n_layer'
name_var2 = 'hid_dim'
list_var1 = [1, 2, 3]
list_var2 = [500, 300]

for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1) # setattr 은 args.name_var1 = var1 이라고 하는 것과 동일 
        setattr(args, name_var2, var2) # 위에서 값을 받아오므로 사용자의 실수를 줄여줄 수 있는 느낌 ! 
        print(args)

        setting, result = experiment(partition, deepcopy(args))
        save_exp_result(setting, result)

```

`!ls results` 을 통해 파일이 잘 생성되었는지를 확인할 수 있다. 

`!cat results/exp1_n_layer_hid_dim-41b634.json` 를 하면 저장된 값들을 불러올 수 있다. 

{"train_losses": [1.7552594508335089, 1.5035573031492293, 1.399491632819935, 1.3134926542354997, 1.246969583687509, 1.189121641930501, 1.1298883709178609, 1.0882677378927825, 1.0231012895608405, 0.978762801285762], 

"val_losses": [1.670531678199768, 1.547693169116974, 1.530729103088379, 1.5106648325920105, 1.7050195932388306, 1.4638991713523866, 1.4923243045806884, 1.4569032311439514, 1.5163418173789978, 1.5914841771125794], 

"train_accs": [37.3225, 46.01, 49.93, 53.055, 55.1475, 57.57, 59.64, 61.3175, 63.5725, 65.0825], 

"val_accs": [40.28, 44.13, 46.26, 47.4, 42.84, 49.12, 48.65, 50.43, 49.2, 49.72], 

"train_acc": 65.0825, "val_acc": 49.72, "test_acc": 50.12, 

"exp_name": "exp1_n_layer_hid_dim", "in_dim": 3072, "out_dim": 10, "hid_dim": 300, "act": "relu", "dropout": 0.2, "use_bn": true, "l2": 1e-05, "use_xavier": true, "optim": "RMSprop", "lr": 0.0015, "train_batch_size": 256, "n_layer": 3}

이렇게, 실험 결과를 하드디스크에 저장할 수 있게 되었다. 

### visualization

이러한 방식으로 여러 실험을 진행한 후, 우리가 변경한 변수들에 대한 acc, loss의 변화 등을 시각화해보자. 

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = load_exp_result('exp1') # 위에서 만들었던 함수를 사용해서 dataframe으로 저장해준다 

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 6)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.barplot(x='n_layer', y='train_acc', hue='hid_dim', data=df, ax=ax[0])
sns.barplot(x='n_layer', y='val_acc', hue='hid_dim', data=df, ax=ax[1])
sns.barplot(x='n_layer', y='test_acc', hue='hid_dim', data=df, ax=ax[2])

```
<img width="1229" height="525" alt="Image" src="https://github.com/user-attachments/assets/2d156856-176b-4e84-b5b2-ca8787d13454" />



```python
var1 = 'n_layer'
var2 = 'hid_dim'

df = load_exp_result('exp1')
list_v1 = df[var1].unique()
list_v2 = df[var2].unique()
list_data = []

for value1 in list_v1:
    for value2 in list_v2:
        row = df.loc[df[var1]==value1]
        row = row.loc[df[var2]==value2]

        train_losses = list(row.train_losses)[0]
        val_losses = list(row.val_losses)[0]

        for epoch, train_loss in enumerate(train_losses):
            list_data.append({'type':'train', 'loss':train_loss, 'epoch':epoch, var1:value1, var2:value2})
        for epoch, val_loss in enumerate(val_losses):
            list_data.append({'type':'val', 'loss':val_loss, 'epoch':epoch, var1:value1, var2:value2})

df = pd.DataFrame(list_data)
g = sns.FacetGrid(df, row=var2, col=var1, hue='type', margin_titles=True, sharey=False)
g = g.map(plt.plot, 'epoch', 'loss', marker='.')
g.add_legend()
g.fig.suptitle('Train loss vs Val loss')
plt.subplots_adjust(top=0.89)
```
<img width="974" height="592" alt="Image" src="https://github.com/user-attachments/assets/3675fcc0-c90a-4069-a8b7-ddb7de16b13f" />




```python
var1 = 'n_layer'
var2 = 'hid_dim'

df = load_exp_result('exp1')
list_v1 = df[var1].unique()
list_v2 = df[var2].unique()
list_data = []

for value1 in list_v1:
    for value2 in list_v2:
        row = df.loc[df[var1]==value1]
        row = row.loc[df[var2]==value2]

        train_accs = list(row.train_accs)[0]
        val_accs = list(row.val_accs)[0]
        test_acc = list(row.test_acc)[0]

        for epoch, train_acc in enumerate(train_accs):
            list_data.append({'type':'train', 'Acc':train_acc, 'test_acc':test_acc, 'epoch':epoch, var1:value1, var2:value2})
        for epoch, val_acc in enumerate(val_accs):
            list_data.append({'type':'val', 'Acc':val_acc, 'test_acc':test_acc, 'epoch':epoch, var1:value1, var2:value2})

df = pd.DataFrame(list_data)
g = sns.FacetGrid(df, row=var2, col=var1, hue='type', margin_titles=True, sharey=False)
g = g.map(plt.plot, 'epoch', 'Acc', marker='.')

def show_acc(x, y, metric, **kwargs):
    plt.scatter(x, y, alpha=0.3, s=1)
    metric = "Test Acc: {:1.3f}".format(list(metric.values)[0])
    plt.text(0.05, 0.95, metric,  horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='yellow', alpha=0.5, boxstyle="round,pad=0.1"))
g = g.map(show_acc, 'epoch', 'Acc', 'test_acc')

g.add_legend()
g.fig.suptitle('Train Accuracy vs Val Accuracy')

plt.subplots_adjust(top=0.89)
```

<img width="974" height="592" alt="Image" src="https://github.com/user-attachments/assets/af47f0a0-4b41-4ca9-84af-e3747f7fb74a" />