---
layout: default
title: "Pytorch Regression"
permalink: /dl/sa/standalone-03/
subtitle: ML Basic 
use_math : true
parent: deep-learning
---

# [Standalone DL] #5 Lab - Regression with Pytorch
## Data Generation

이번에는 x 2차원, y 1차원으로 가보자. 다음 함수를 따르는 x와 y를 생각해보자 

$$ e \sim \mathcal{N} (0, 0.5) $$
$$y = \ 2 sin(x_1) + log({1 \over 2}x_2^2) + e$$

를 따르는 데이터셋 2400개를 수집했다고 생각해보자.

**Data Set**  
$$X_{train} \in \mathcal{R}^{1600 \times 2}, Y_{train} \in \mathcal{R}^{1600}$$  
$$X_{val} \in \mathcal{R}^{400 \times 2}, Y_{val} \in \mathcal{R}^{400}$$  
$$X_{test} \in \mathcal{R}^{400 \times 2}, Y_{test} \in \mathcal{R}^{400}$$

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ====== Generating Dataset ====== #
num_data = 2400
x1 = np.random.rand(num_data) * 10 # 0~1 unif에서 num_data 개수만큼 뽑는 함수 
x2 = np.random.rand(num_data) * 10
e = np.random.normal(0, 0.5, num_data) # µ=0, sd=0,5인 gaussian dist 
X = np.array([x1, x2]).T
y = 2*np.sin(x1) + np.log(0.5*x2**2) + e

# ====== Split Dataset into Train, Validation, Test ======#
train_X, train_y = X[:1600, :], y[:1600]
val_X, val_y = X[1600:2000, :], y[1600:2000]
test_X, test_y = X[2000:, :], y[2000:]

# ====== Visualize Each Dataset ====== #
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(train_X[:, 0], train_X[:, 1], train_y, c=train_y, cmap='jet')

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('Train Set Distribution')
ax1.set_zlim(-10, 6)
ax1.view_init(40, -60)
ax1.invert_xaxis()

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(val_X[:, 0], val_X[:, 1], val_y, c=val_y, cmap='jet')

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('Validation Set Distribution')
ax2.set_zlim(-10, 6)
ax2.view_init(40, -60)
ax2.invert_xaxis()

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('Test Set Distribution')
ax3.set_zlim(-10, 6)
ax3.view_init(40, -60)
ax3.invert_xaxis()

plt.show()
```

<img width="971" height="341" alt="Image" src="https://github.com/user-attachments/assets/5f3b80ef-465a-4d4e-91d5-29058386a4c2" />

`train set` : 학습할 때 사용하는 데이터셋 (업데이트까지 완료) 

`validation set` : train set의 loss가 줄다 보면 validation set의 loss는 증가하게 될 수도 있음. 즉,  validation에서는 잘 못하게 되는 순간이 발생할 수 있음 (=overfitting) - 이를 체크하기 위해 사용 

- train 한 번 당 validation 한 번 , train 10번 당 validation 한 번 이렇게 마음대로 진행할 수 있음

`test set` : 평가용 데이터셋 

<br />   

## Hypothesis Define

2차원짜리 input에 W를 곱하고 bias를 더해서 1차원으로 변경해주는 작업 진행 ! 

**Linear Model**

$$
H = XW + b \quad \left( W \in \mathbb{R}^{2 \times 1},\ b \in \mathbb{R}^{1},\ H \in \mathbb{R}^{N \times 1} \right)
$$

Let \( \text{ReLU}(X) = \max(X, 0) \)

$$
h = \text{ReLU}(X W_1 + b_1) \quad \left( W_1 \in \mathbb{R}^{2 \times 200},\ b_1 \in \mathbb{R}^{200},\ h \in \mathbb{R}^{N \times 200} \right)
$$

$$
H = h W_2 + b_2 \quad \left( W_2 \in \mathbb{R}^{200 \times 1},\ b_2 \in \mathbb{R}^{1},\ H \in \mathbb{R}^{N \times 1} \right)
$$

**Mean Absolute Error**

$$
\text{MAE}(Y_{\text{true}}, Y_{\text{predict}}) = \sum_{i} \left| y_{\text{true}}^{(i)} - y_{\text{predict}}^{(i)} \right|
$$

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module): # nn.Module 안에 더티한 작업들이 다 있고, 우린 그걸 그냥 상속해오겠다 
    def __init__(self):
        super(LinearModel, self).__init__() # 클래스 쓰는 규칙 
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):
        return self.linear(x) # x가 x 자리에 들어가서 w랑 곱해지고 b랑 더해져서 값 완성됨 

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x): # 인공신경망의 가장 기초 형태 !! 
        x = self.linear1(x) # Wx+b
        x = self.relu(x) # 선형적이지 않은 어떠한 함수 거치기 
        x = self.linear2(x) # Wx+b 
        return x

lm = LinearModel()
print(lm.linear.weight) # tensor([[-0.1405,  0.1758]], requires_grad=True)
```

nn.Module이 알아서 잘 처리해주고 있는 과정을 확인해볼 수 있다 

인공신경망의 가장 기초적인 형태를 여기서 볼 수 있었는데, 

실제로 깊은 인공신경망, deep-learning의 진행은 `self.relu(x)`와 `self.linear2(x)`의 반복이다 

<br />   

## Cost Function Define

$$

MAE(Y*_{true}, Y_*{predict}) = \sum*_{i} | \ y_*{true}^{(i)} - y_{predict}^{(i)} \ | 

$$

```python
reg_loss = nn.MSELoss()

test_pred_y = torch.Tensor([0,0,0,0])
test_true_y = torch.Tensor([0,1,0,1])

print(reg_loss(test_pred_y, test_true_y))
print(reg_loss(test_true_y, test_true_y))
```

<br />    

## Train & Evaluation

```python
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

# ====== Construct Model ====== #
# model = LinearModel()
# print(model.linear.weight)
# print(model.linear.bias)

model = MLPModel() 
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))) # 복잡해보이지만 간단히 모델 내에 학습을 당할 파라미터 수를 카운팅하는 코드입니다.

# ===== Construct Optimizer ====== #
lr = 0.005 
optimizer = optim.SGD(model.parameters(), lr=lr)

list_epoch = []
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []

epoch = 4000 .
for i in range(epoch):

    # ====== Train ====== #
    model.train() # 모델 학습 모드 
    optimizer.zero_grad()

    input_x = torch.Tensor(train_X)
    true_y = torch.Tensor(train_y)
    pred_y = model(input_x)

    loss = reg_loss(pred_y.squeeze(), true_y)
    loss.backward() # 백워드로 그라디언트 구해주기 
    optimizer.step() # 그라디언트 바탕으로 파라미터 업뎃해주기 
    list_epoch.append(i)
    list_train_loss.append(loss.detach().numpy())

    # ====== Validation ====== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X)
    true_y = torch.Tensor(val_y)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.detach().numpy())

    # ====== Evaluation ======= #
    if i % 200 == 0: # 200번에 한 번씩 평가하도록 함 

        # ====== Calculate MAE ====== #
        model.eval()
        optimizer.zero_grad()
        input_x = torch.Tensor(test_X)
        true_y = torch.Tensor(test_y)
        pred_y = model(input_x).detach().numpy()
        mae = mean_absolute_error(true_y, pred_y) 
        list_mae_epoch.append(i)

        fig = plt.figure(figsize=(15,5))

        # ====== True Y Scattering ====== #
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('y')
        ax1.set_zlim(-10, 6)
        ax1.view_init(40, -40)
        ax1.set_title('True test y')
        ax1.invert_xaxis()

        # ====== Predicted Y Scattering ====== #
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter(test_X[:, 0], test_X[:, 1], pred_y, c=pred_y[:,0], cmap='jet')

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('y')
        ax2.set_zlim(-10, 6)
        ax2.view_init(40, -40)
        ax2.set_title('Predicted test y')
        ax2.invert_xaxis()

        # ====== Just for Visualizaing with High Resolution ====== #
        input_x = torch.Tensor(train_X)
        pred_y = model(input_x).detach().numpy()

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.scatter(train_X[:, 0], train_X[:, 1], pred_y, c=pred_y[:,0], cmap='jet')

        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_zlabel('y')
        ax3.set_zlim(-10, 6)
        ax3.view_init(40, -40)
        ax3.set_title('Predicted train y')
        ax3.invert_xaxis()

        plt.show()
        print(i, loss)
```

그리고 evaluation을 해보자 !
<img width="882" height="333" alt="Image" src="https://github.com/user-attachments/assets/30ebc09c-a195-4f84-909c-864f0a85b216" />

- loss를 저장하고 출력해서 그래프를 그려보는 행위는 매우 중요함 ! (학습 진행 상황을 계속 파악하는 것이 중요함)

