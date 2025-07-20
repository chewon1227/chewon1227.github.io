---
layout: default
title: "Binary / Multi-Label Classification and Lab"
permalink: /dl/sa/standalone-04/
subtitle: ML Basic 
use_math : true
parent: deep-learning
---

# [Standalone DL] #6 Lecture - Binary / Multi-Label Classification

## Binary Classification

### Old Hypothesis - linear regression

다음 세 가지 요소가 가장 핵심이다. 

- hypothesis (우리의 model)
- cost (잘 예측하면 할수록 값이 줄어드는)
- optimization (cost기반으로 hypothesis의 parameter들을 업데이트하는 과정인)

linear regression의 한계점 

- pass (1) / fail (0) 의 binary 분류에 대해서, x 값이 엄청 큰데 여전히 pass (1) 이라는 것 때문에 변형이 생길 수 있음
- 그 변형 때문에 앞부분에서는 0보다 작은 값이 나올 수도 있음
- 즉, linear regression을 binary에 가져갈 수 없음

⇒ linear regression에 적절한 함수를 덮어씌우면 될 것 같다고 생각함 ! 

$$

H(x) = G(Wx+b)

$$

<br />    

### New Hypothesis - Sigmoid (logistic) Function

<img width="372" height="253" alt="Image" src="https://github.com/user-attachments/assets/9c641a9f-9a7e-4a8b-9f7f-eb00f698fb3e" />

$$
G(z) = \frac{1}{1 + e^{-z}}
H(X) = \frac{1}{1 + e^{WX}}
$$

**cost problem** 

- 원래 쓰던 cost function을 쓰면 모양이 이상하게 나옴. gradient=0인 포인트 (local minimum) 들이 중간중간 많이 끼게 됨.
- step을 밟아나갈 수 없게 됨

**cross entropy**

$$
H(P, Q) = - \sum P(x) \log(Q(x))
$$

- entropy : 시스템이 얼마나 불안정한가.
- 여기에서 cross entropy : 두 개의 확률분포의 차이
- P(x) : 실제 확률 / Q(x) : 예측한 확률

**cost function**

$$
\text{cost}(W) = \frac{1}{m} \sum_{i=1}^{n} c(H(x^{(i)}), y^{(i)})
$$

$$
c(H(x), y) =
\begin{cases}

- \log(H(x)) & \text{if } y = 1 \\
- \log(1 - H(x)) & \text{if } y = 0
\end{cases}

$$

- pass 한 경우 y=1, fail한 경우를 y=0이라고 하고, study hour이라는 인풋을 받는다고 하자
- pass할 것이라 생각했는데 fail한 경우 / fail할 것이라 생각했는데 pass한 경우 (큰값) 를 생각해볼 수 있다
- 실제로 pass 할 것이라 생각했는데 pass한 경우 / fail할 것이라 생각했는데 fail한 경우 (작은 값) 를 생각해볼 수 있다

<br />   

<br />   

## Multinomial Classification

### hypothesis

binary classification에서는 선 하나를 그으면 그룹을 나눠서 깔끔하게 해결할 수 있었다.

그러나 multinomial classification에서는 선 하나로는 불가능해짐. 또한 그룹 자체가 선으로 분리되지 않는 그룹일 수 있음 !! 

<br />
![Image](https://github.com/user-attachments/assets/51d5fd7f-4142-4aa5-b5f3-118459da61d9)

<br />  

### Softmax Function

$$
\text{Softmax}(\hat{y}_i) = \frac{e^{\hat{y}_i}}{\sum_j e^{\hat{y}_j}}
$$

continuous한 값을 각각 클래스에 대한 확률로 변환할 수 있게 됨 ! 

<br />  

### cost function

예측한 값과 실제 값을 비교해서 나온 값.

cross-entropy 개념을 가져온다 ! 실제와 예측이 얼마나 다른가를 ‘불확실성’을 가지고 생각한다 

`pred_y` (softmax를 통과한 값들) 과 `true_y` (실제 맞으면 1, 아니면 0) 을 비교한다 . 

## Pytorch Classification

### classification

**Data Set**  
$$

X_{train} \in \mathcal{R}^{8000 \times 2}, Y_{train} \in \mathcal{Z}^{8000}

X_{val} \in \mathcal{R}^{1000 \times 2}, Y_{val} \in \mathcal{Z}^{1000}

X_{test} \in \mathcal{R}^{1000 \times 2}, Y_{test} \in \mathcal{Z}^{1000}

$$
<img width="1233" height="470" alt="Image" src="https://github.com/user-attachments/assets/1aa60f54-54c9-44ec-be1d-cdc13812cf08" />

<br />   

### Hypothesis Define

**Multi-Label Logistic Model**   
$$z = \ XW + b \ \ ( W \in \mathcal{R}^{2 \times 3}, b \in \mathcal{R}^{3}, z \in \mathcal{R}^{N \times 3}$$  
$$H = \ softmax(z) \ \ (  H \in \mathcal{R}^{N \times 3})$$  

**MLP Model**
$$Let \ relu(X) = \ max(X, 0)$$  

$$h = \ relu(X W_1 + b_1) \ \  ( W_1 \in \mathcal{R}^{2 \times 200}, b_1 \in \mathcal{R}^{200}, h \in \mathcal{R}^{N \times 200}$$  

$$z = \ h W_2 + b_2  \ \  ( W_2 \in \mathcal{R}^{200 \times 3}, b_2 \in \mathcal{R}^{3}, z \in \mathcal{R}^{N  \times 3})$$  

$$H = \ softmax(z) \ \ ( H \in \mathcal{R}^{N \times 3})$$  

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self): 
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=3, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        return x

    
class MLPModel(nn.Module):
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

모델 내에서 굳이 softmax를 사용할 필요가 없음 (CrossEntropyLoss 함수에 softmax 함수가 포함되어 있기 때문임 !!) 

<br />    

### Cost (=Loss) Function Define

pytorch 안의 nn 에는 다양한 loss function이 이미 구현되어 있음. 

```python
cls_loss = nn.CrossEntropyLoss()

test_pred_y = torch.Tensor([[2,0.1],[0,1]])
test_true_y1 = torch.Tensor([1,0]).long()
test_true_y2 = torch.Tensor([0,1]).long()

print(cls_loss(test_pred_y, test_true_y1)) # tensor(1.6763)
print(cls_loss(test_pred_y, test_true_y2)) # tensor(0.2263)
```

<br />  

### Train & Evaluation

```python
import torch.optim as optim
from sklearn.metrics import accuracy_score

# ====== Construct Model ====== #
model = LinearModel()
print(model.linear.weight)
print(model.linear.bias)

model = MLPModel() # Model 생성 
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))) # 복잡해보이지만 간단히 모델 내에 학습을 당할 파라미터 수를 카운팅하는 코드입니다.

# ===== Construct Optimizer ====== #
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr) # Optimizer를 생성해줍니다.

list_epoch = [] 
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 4000 # 학습 횟수 지정
for i in range(epoch):
    
    # ====== Train ====== #
    model.train() # train 모드로 세팅 
    optimizer.zero_grad() # 그라디언트 초기화 
    
    input_x = torch.Tensor(train_X)
    true_y = torch.Tensor(train_y).long()
    pred_y = model(input_x)
    
    loss = cls_loss(pred_y.squeeze(), true_y)
    loss.backward() # backward()로 그라디언트 구하기 
    optimizer.step() # step()으로 파라미터 업뎃  
    list_epoch.append(i)
    list_train_loss.append(loss.detach().numpy())
    
    
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

<img width="716" height="293" alt="Image" src="https://github.com/user-attachments/assets/df2a49bd-6554-46e5-a04d-353ff4ce54c1" />

마지막 에폭의 결과는 다음과 같다 ! 학습이 진행될수록 Prediction on train set의 결과가 매우 괜찮아지는 것을 볼 수 있다 (사실 처음 이후로 다 나쁘지 않다) 

<br />    

### Report Experiment

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

<img width="1233" height="470" alt="Image" src="https://github.com/user-attachments/assets/33b15818-4c4f-4803-a24e-e5f6e2c376d6" />


