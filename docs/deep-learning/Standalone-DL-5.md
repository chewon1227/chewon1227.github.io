---
layout: default
title: "05 DL and MLP Basic "
parent: deep-learning
permalink: /dl/sa/standalone-05/
use_math : true
---
# [Standalone DL] 05 - #8-9 History of DL / MLP Basic 

## Modeling Neuron

### Basic

<img width="478" height="288" alt="Image" src="https://github.com/user-attachments/assets/7b7982b2-70ab-48fe-8dcc-b9c3db5a9676" />

뉴런을 수학적으로 모델링함 

- 들어온 신호들에 적절한 `weight`들을 잘 곱해줌
- 들어온 신호들의 합이 작아서 역치를 넘지 않으면, 이 뉴런은 다음 뉴런으로 신호를 전달하지 않음 (`activation function`)
- 그렇다면 이 `activation  function`은 non-linear한 형태를 띄고 있다 !!

<br />    

### And/Or Problem

간단한 신경망을 만들고 이를 이용해서 문제를 풀어보자
<img width="404" height="303" alt="Image" src="https://github.com/user-attachments/assets/43f08efc-d7a2-4da1-8c3b-79deccb986c9" />
- 두 가지 인풋을 받아서 0 또는 1의 아웃풋을 내놓는 형태
- and / or problem은 쉽게 풀림 !
- 그러나 XOR problem을 풀 수 없음

### XOR Problem by Multilayer Perceptron

<img width="500" height="275" alt="Image" src="https://github.com/user-attachments/assets/add03224-3610-4d1c-bdce-e71facb9e0be" />

- 중간에 레이어가 한 개만 있는 경우 절대 XOR 문제를 풀 수 없다는 것을 수학적으로 증명
- Multilayer perceptron (hidden-layer이 1개 이상 있는!) 을 통해서 풀 수 있다는 것을 수학적으로 증명
- 근데 weight, bias가 너무 많이 생김. training을 어떻게 할 것인가?

### Backpropagation

<img width="420" height="318" alt="Image" src="https://github.com/user-attachments/assets/7465dc9f-9f9f-432c-9245-0b163f51a2c5" />


- 수많은 weight, bias 등의 parameter들을 계산해서 업데이트할 수 있는 training 알고리즘이 나옴 !
- 이제 MLP를 training할 수 있게 됨

<br /><br />    

## Convolutional Neural Networks

### Logic

고양이의 뇌 신호를 분석함 

- 간단한 이미지를 보여줌 (선이 왔다갔다 하거나, 선의 기울기가 변화하거나, 원 사이즈가 변화하거나)
- 고양이의 뉴런이 선택적으로 활성화됨
- 우리의 뇌는 이미지를 볼 때 전체를 하나로 보는 것이 아니라, 부분 부분을 각각 처리한 후 뒤에 가서 한 번에 합쳐서 처리하는 것 아닌가? 라는 생각을 하게 됨

### CNN
<img width="676" height="272" alt="Image" src="https://github.com/user-attachments/assets/596e2940-81b1-4d60-8407-f3286943b331" />
- 점점 뒤로 갈수록 전체 이미지를 보는 형식
- ImageNet Classification에서, 이제 사람의 오류율(3%)와 유사한 ResNet 구조까지 나오게 됨

<br /><br />

# MLP Regression with Pytorch 

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim): # in_dim, out_dim, hid_dim을 넣어준 이유: explicit하게 차원 설정하지 말고 변수로 넣고 싶을 경우 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim) # 2, 200
        self.linear2 = nn.Linear(hid_dim, out_dim) # 200, 1 
        self.relu = nn.ReLU() # ReLU activation 가져오기 

    def forward(self, x):
    # forwa
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

```python
cls_loss = nn.CrossEntropyLoss()
```

```python
import torch.optim as optim
from sklearn.metrics import accuracy_score

# ====== Construct Model ====== #
model = LinearModel()
#model = MLPModel(in_features=2, out_features=1, hidden_features=200)
model = MLPModel(in_features=2, out_features=3, hidden_features=200)

# ===== Construct Optimizer ====== #
lr = 0.005
ptimizer = optim.SGD(model.parameters(), lr=lr) 

list_epoch = []
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 4000 
for i in range(epoch):

    # ====== Train ====== #
    model.train() # train mode 
    optimizer.zero_grad() # gradient initialize 
    
    for input_X, true_y in train_loader:
		    input_X = input_X.squeeze() # input: torch.tensor 되어있음. 1*28*28..처럼 되어있는데 squeeze를 통해 1을 없앰 
		    input_X = input_X.view(-1, 784) # 784x1 차원으로 바꿔주는 함수 
		    pred_y = model(input_X)
		    
		    loss = cls_loss(pred_y.squeeze(), true_y) 
		    loss.backward()
		    optimizer.step()
		    train_loss += loss.detach().numpy()
		 train_loss = train_loss / len(train_loader)
		 list_train_loss.append(train_loss)
		 list_epoch.append(i)
		 
    # ====== Validation ====== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X)
    true_y = torch.Tensor(val_y).long()
    pred_y = model(input_x)
    loss = cls_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.detach().numpy())

    # ====== Evaluation ======= #
    if i % 200 == 0:

        # ====== Calculate Accuracy ====== #
        model.eval()
        optimizer.zero_grad()
        input_x = torch.Tensor(test_X)
        true_y = torch.Tensor(test_y)
        pred_y = model(input_x).detach().max(dim=1)[1].numpy()
        acc = accuracy_score(true_y, pred_y) 
        list_acc.append(acc)
        list_acc_epoch.append(i)

        fig = plt.figure(figsize=(15,5))

        # ====== True Y Scattering ====== #
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(test_X[:, 0], test_X[:, 1], c=test_y)
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title('True test y')

        # ====== Predicted Y Scattering ====== #
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(test_X[:, 0], test_X[:, 1], c=pred_y)
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title('Predicted test y')

        # ====== Just for Visualizaing with High Resolution ====== #
        input_x = torch.Tensor(train_X)
        pred_y = model(input_x).detach().max(dim=1)[1].numpy()

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(train_X[:, 0], train_X[:, 1], c=pred_y)
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_title('Prediction on train set')

        plt.show()
        print('Epoch: ', i,  'Accuracy: ', acc*100, '%')
```

## Loss

```python
fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_acc_epoch, list_acc, marker='x', label='Accuracy metric')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Acc')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs Accuracy')

plt.show()
```