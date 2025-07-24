---
layout: default
title: "02 Linear Regression and Lab"
permalink: /dl/sa/standalone-02/
subtitle: Linear Regression 
use_math : true
parent: deep-learning
---

# [Standalone DL] #3-4 - 02 Linear Regression and Practice 

## Concept

<img width="518" height="251" alt="Image" src="https://github.com/user-attachments/assets/6002ee3b-8c5e-4f68-8a4f-96077267202b" />
<img width="494" height="318" alt="Image" src="https://github.com/user-attachments/assets/5c492fb4-f90a-488b-bd35-c02dc3a914fa" />

목적 : 데이터를 가장 잘 설명하는 line hypothesis를 찾는 것 

- 무엇이 ‘좋은’ 설명인지 어떻게 판단하는가?
- 이것을 정의해야 W, b를 조절할 수 있을 것

<br />

## Cost Function

= Loss function 과 동일한 단어이다. 

목표 : training data와 line을 fit 하는 과정 ! $ H(x) - y $ 

$$
cost(W, b) = \frac{1}{m} \sum_{i=1}^{m} \left( H\left(x^{(i)}\right) - y^{(i)} \right)^2
$$

- W, b에 의해 cost 값이 변동하게 될 것
- 이 cost를 minimize하는 W, b를 찾는 것이 목표이다 !!

<br />
<br />

# Minimizing Cost

## Simplified hypothesis

<img width="368" height="269" alt="Image" src="https://github.com/user-attachments/assets/ef73966d-161c-4bda-ad4c-9e171e3ba747" />
- x - y로 매칭되는 데이터 예시에 대해, 직접 계산해보면 다음과 같이 값이 얻어지게 된다
- 그러나 차원이 복잡해지거나, 데이터 양이 많아지면 이렇게 그래프를 그릴 수 없게 됨 ! → 잘 모르는 곳에서 학습을 하는 느낌이 들 것
- 이것을 어떻게 ‘algorithm’틱하게 줄일 수 있을 것인가?

<br />

## Gradient Descent Algorithm

### Method 
- start with initial guesses (0 or any value)
- parameter 바꿀 때마다 gradient 선정 (cost를 가장 줄일 수 있는 것으로) 하는 작업 반복
- local minimum에 도달할때까지 . . . (근데 이걸 명확하게 알 수 있나?) 
→ optimize 문제가 생길 수 있다는 문제에 대비하여 . . . modern한 optimizer을 사용하게 될 것

<br />

### Feature 

$$
W := W - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( W x^{(i)} - y^{(i)} \right) x^{(i)}
$$

<img width="489" height="294" alt="Image" src="https://github.com/user-attachments/assets/9c22b0ee-c1eb-4cff-bdac-864c04a736ae" />

w의 시작점에 따라 종착점이 달라질 수 있음 

<br />
<br />

# 3 variables

## Concepts

regression using three inputs ($ x_1, x_2, x_3$ )

$$
H(x_1, x_2, x_3) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
$$

$$
cost(W, b) = \frac{1}{m} \sum_{i=1}^{m} \left( H(x_1^{(i)}, x_2^{(i)}, x_3^{(i)}) - y^{(i)} \right)^2
$$

<br />

## Hypothesis Using Matrix
<img width="420" height="171" alt="Image" src="https://github.com/user-attachments/assets/e450a800-0dcf-456e-bb21-491f8f59f6bc" />
- 예측해야 할 것이 1개인 경우 

<img width="465" height="147" alt="Image" src="https://github.com/user-attachments/assets/e17d6903-be9e-449f-a831-d742c73353fa" />
- 예측해야 할 것이 2개인 경우 


<br /><br />

## Code Basic

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