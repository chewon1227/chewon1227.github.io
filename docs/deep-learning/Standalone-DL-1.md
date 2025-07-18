---
layout: post
title: "[Standalone DL] #2 lecture - ML Basic"
permalink: /dl/standalone-01/
subtitle: ML Basic 
use_math : true
parent: deep-learning
---

# What is ML?

## Definition

- A Field of sstudy that gives computer the ability to learn without being explicitly programmed
- 직접 rule-base (수백개의 if-else 문 등) 없이 implicit하게 짜도 스스로 학습을 해낸다

<br />

<br />

# Categories

## ML and others
<img width="702" height="298" alt="Image" src="https://github.com/user-attachments/assets/05d66f84-8ab8-4c3d-b5e3-c96c0d1d5cdb" />
- 학습법 : supervised, unsupervised, reinforcement learning
    - `unsupervised learning`
        - x만 있는 경우 활용할 수 있는 학습 방법. 비슷한 것끼리 묶은 후 사람이 라벨링을 해주는 방식
    - `reinforcement learning`
        - x, y가 주어지지 않음
        - 현재 자신의 state + environment가 input, 어떠한 action이 output → state update → …
        - reward function을 통해, state가 나아진다면 action에 대해 reward 제공
- output space : continuous, discrete

<br />

### Regression Problem
<img width="618" height="334" alt="Image" src="https://github.com/user-attachments/assets/0eda509f-4d9d-4669-9318-3b818df22de1" />
<img width="620" height="266" alt="Image" src="https://github.com/user-attachments/assets/3ab996eb-af44-4f2f-8317-67553dacc0c7" />
목표 : 다양한 function들을 try하며, 가장 잘 fitting하는 것을 찾는 것 

- nonlinear하게 데이터가 분포되어 있다면 - nonlinear regression
- nonlinear - ml에서도, dl에서도 가능함. ml에서는 한계가 있기에 dl로 가는 추세

<br />

### Classification Problem
<img width="654" height="291" alt="Image" src="https://github.com/user-attachments/assets/a01ac413-5b38-4762-8148-518586ac4fe7" />
<img width="642" height="346" alt="Image" src="https://github.com/user-attachments/assets/b6683d64-e2ef-43fb-a0a9-7e0f1587d5ce" />
목표 : 다양한 데이터들을 특정한 기준을 가지고 분류하는 것 

- decision boundary를 학습하게 됨
- linear regression이라면, 구분할 때 0.5=0.5로 확률이 같은 지점들을 모아보면 line이 됨

<br />

### Clustering Problem
<img width="660" height="252" alt="Image" src="https://github.com/user-attachments/assets/0e7ade3d-4648-4b77-ab70-906ee5d6da0f" />
<img width="593" height="222" alt="Image" src="https://github.com/user-attachments/assets/15b94cd5-de10-4542-8ba4-0c8511fb7a9a" />
목표 : 알아서 grouping이 되도록 하는 것 

- instance 내 similarity, distance를 정의해서 비슷한 애들끼리 모은다는 점이 핵심
- dl을 통해 100-300차원의 벡터로 뽑아서 similarity 등을 계산함
- 방법은 다양함 (k-Means, EM 등) . dl로 가면 더 복잡한 분포들에 대해서 더 깔끔하게 처리 가능함

<br />  

### Dimensionality Reduction Problem

<img width="613" height="292" alt="Image" src="https://github.com/user-attachments/assets/555c184e-f7e0-4039-9a83-76643195729f" />
차원의 저주

- 고차원에서 계산되는 유클리안 distance는 점점 의미가 없어짐
- 데이터의 정보를 손실하지 않기 위해 저차원으로 보내려는, dimension을 낮추려는 노력을 함

<br />  

<br />  

## ML Problems

<img width="590" height="302" alt="Image" src="https://github.com/user-attachments/assets/31af10b2-9e2b-4351-96b3-99668828705d" />


### **Feature & Data Representation**

**Case 1.  3차원**

<img width="475" height="130" alt="Image" src="https://github.com/user-attachments/assets/fcff65c7-0b0d-4ac0-850d-a0f6b9b96ffc" />

**Case 2. 784차원**
<img width="423" height="177" alt="Image" src="https://github.com/user-attachments/assets/6a38af6d-18ba-4938-8dfe-834d6eddef0d" />
- 784차원 상에서의 유클리드 distance 값을 계산해볼 수 있음 - but 고차원의 저주로 인해 5-5 사이의 거리보다 5-0 사이의 거리가 더 적게 될 수도 있음
- 이때, dimensionality reduction을 통해 x를 784차원 → 3차원으로 보내는 알고리즘 을 만든다 !