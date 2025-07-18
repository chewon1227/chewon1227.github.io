---
layout: default
title: "[Standalone DL] #3 lecture - Linear Regression"
permalink: /dl/standalone-01/
subtitle: ML Basic 
use_math : true
parent: deep-learning
---

# Linear Regression

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


