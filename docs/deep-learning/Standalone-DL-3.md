---
layout: default
title: "[Standalone DL] #4 Lab - Linear Regression Practice"
permalink: /dl/standalone-03/
subtitle: ML Basic 
use_math : true
parent: deep-learning
---

## Basic

x, y를 넣고, matplotlib를 통해 그림을 그린다. 

```python
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = [1, 1, 2, 4, 5, 7, 8, 9, 9, 10]

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.show()
```

<img width="543" height="413" alt="Image" src="https://github.com/user-attachments/assets/72042e74-7829-497f-817b-8404c936cbc2" />

<br />

## Make Function

```python
class H():
	def __init__(self, w):
		self.w=w
	
	def forward(self, x):
		return self.w *x # linear 함수를 만들어낸 것 
```

```python
h = H(4) # f(x)=4x 가 만들어진 것. w=4
cost(h,X,Y) # 222.2 
```

## Cost Function

```python
# ver 1 : cost function 안에서 H(x)를 계산해야 함. h.forward(X[i])
def cost(h, X, Y):
    error = 0
    for i in range(len(X)):
        error += (h.forward(X[i]) - Y[i])**2
    error = error / len(X)
    return error

h = H(4) 
print('cost value when w = 4 :', cost(h, X, Y)) 

# ver 2 : 모델 예측값과 실제 값의 리스트를 받는 형태 - 훨씬 간단함 
def better_cost(pred_y, true_y):  
    error = 0
    for i in range(len(X)):
        error += (pred_y[i] - true_y[i])**2
    error = error / len(X)
    return error

pred_y = [ h.forward(X[i]) for i in range(len(X)) ] 
print('cost value with better code structure :', better_cost(pred_y, Y))
```

```python
list_w = []
list_c = []
for i in range(-20, 20):
    w = i * 0.5
    h = H(w)
    c = cost(h, X, Y)
    list_w.append(w)
    list_c.append(c)
    
plt.figure(figsize=(10,5))
plt.xlabel('w')
plt.ylabel('cost')
plt.scatter(list_w, list_c, s=3)
```

range(-20, 20)일 때는 다음과 같은 결과,
<img width="859" height="448" alt="Image" src="https://github.com/user-attachments/assets/d841227b-981d-4511-928f-9bdfc3e5fba4" />
range(-100, 100)일 때는 다음과 같은 결과가 나온다. 
<img width="868" height="448" alt="Image" src="https://github.com/user-attachments/assets/3076cf91-cb81-4184-ba91-900a78b9c61a" />

<br />

## Gradient

수치학적으로 gradient를 근사한다 

```python
def cal_grad(w, cost): # 여기서 cost는 함수 자체 
    h = H(w)
    cost1 = cost(h, X, Y)
    eps = 0.00001 
    h = H(w+eps) 
    cost2 = cost(h, X, Y)
    dcost = cost2 - cost1
    dw = eps
    grad = dcost / dw
    return grad, (cost1+cost2)*0.5

def cal_grad2(w, cost):
    h = H(w)
    grad = 0
    for i in range(len(X)):
        grad += 2 * (h.forward(X[i]) - Y[i]) * X[i]
    grad = grad / len(X)
    c = cost(h, X, Y)
    return grad, c
```

```python
w = 4
lr = 0.001
print(cal_grad(4, cost)) # 159.028... : w=4일때는 w가 증가할 때 cost 증가한다. 
w = w + lr* (-cal_grad(4,cost)) # -를 취해서
print(w) # 3.84... # w를 낮춘다 ! 
```

스캐터플랏을 그려본다 

```python
w1 = 1.4
w2 = 1.4
lr = 0.01

list_w1 = []
list_c1 = []
list_w2 = []
list_c2 = []

for i in range(100): 
    grad, mean_cost = cal_grad(w1, cost)
    grad2, mean_cost2 = cal_grad2(w2, cost)

    w1 -= lr * grad
    w2 -= lr * grad2
    list_w1.append(w1)
    list_w2.append(w2)
    list_c1.append(mean_cost)
    list_c2.append(mean_cost2)
      
plt.scatter(list_w1, list_c1, label='analytic', marker='*')
plt.scatter(list_w2, list_c2, label='formula')
plt.legend()
```

<img width="547" height="413" alt="Image" src="https://github.com/user-attachments/assets/742e86b2-b840-4f93-b2d9-930c10e650a4" />
수렴하고 있는 것을 알 수 있음 ! 
<img width="547" height="446" alt="Image" src="https://github.com/user-attachments/assets/446eb654-cb5a-41d8-8c2c-97fe4c6a40b1" />
learning_rate를 0.1로 두니 over-shooting 발생 ! (1.2까지 가지않고 이상한 곳에서 왔다갔다 하고 있음) 