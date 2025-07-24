---
layout: default
title: "07 Overfitting and Hyperparameter "
parent: deep-learning
permalink: /dl/sa/standalone-07/
subtitle: Dl and MLP 
use_math : true
---

# [Standalone DL] #13 Lecture - 07 Overfitting, Regularization 

## Prob  1: Overfitting

### Model Capacity

- 높아질수록 더 많은 현상을 설명할 수 있게 된다
- ex. 2차보다 4차함수가 model capacity가 더 높을 수 있음

### Overfitting

- 학습하고 있는 데이터셋에 대해 과하게 학습해서, 좀 특이한 케이스 까지도 완전하게 학습해버린 상태.
- training 세트에만 완벽하게 적응한 상태
- 그렇지만 실제로 사용하는 것은 test set일 것 (혹은 테스트셋에 없는 데이터들) - 이렇게 general한 경우에는 적용이 잘 되지 않을 것

### Risk

**True Risk** 

- 전체 모든 데이터에 대한 risk
- 실제로 계산할 수 없음

**Empirical Risk** 

- 아주 작은 subset으로 계산한 risk
- true risk를 줄이기 위해 이것을 줄여나감 !

training set을 외워버린 것은 아닌지 확인하는 과정이 꼭 필요함 ( = validation set을 통해서 !) 

## Ans : Regularization

패널티를 준다 / 일반화를 한다 

### A1. L2 Regularization

$$

Loss = MSE + R(/lambda)

R(/lambda) = /lambda ||w|| ^2 

$$

- 각 파라미터들이 너무 발산하지 않도록 제한하는 것
- loss가 줄어드는 방향으로 가야 하기 때문 - 만약 w가 너무 커진다면 loss 값 자체도 커질 것이기 때문에 이를 제한할 수 있게 됨 (그 방향으로 가지 않도록!)

왜 오버피팅을 줄여주는가?

<img width="491" height="307" alt="Image" src="https://github.com/user-attachments/assets/e7080e02-94de-429c-aabe-81acd16a7d8c" />

- 학습 과정 = global minimum을 찾아가는 과정. BUT 데이터셋을 바꾸면 global minimum이 다른 곳에 있을 것이기 때문에 제대로 예측이 되지 않을 수 있음
- 이때 Regularization 항을 추가해줄 경우 - global minimum에 빠지지 않고 다른 어느 적절한 곳에 도착하도록 하기 위함임 
→ 다른 데이터에서도 잘 적응력을 가지기 위해
- 수학적 증명 가능 !

<br />     

### A2. Dropout

<img width="532" height="292" alt="Image" src="https://github.com/user-attachments/assets/84d00396-5b60-4c79-a790-8b67a9e96e12" />

p의 확률로 각각의 노드를 꺼버리는 것 

- 너무 복잡하게 예측하지 말고 최대한 간단하게 예측할 수 있도록 학습해라
- 노드가 많으면 학습이 잘 안될 수 있음 (나뉘는 것이니까) - 몇 개는 꺼버리면 효율적으로 학습이 잘 될 수 있음
- 노드가 꺼지고 켜질 때마다 또 새로운 하나의 모델이 되는 것.
- **즉, 복잡한 하나의 모델로 예측하는 것이 아니라, 간단한 여러 종류의 모델로 예측해서 총합하여 사용하는 것 ( ~ 앙상블 기법)**

<br /><br />    

## Prob 2: Gradient Vanishing

<img width="697" height="140" alt="Image" src="https://github.com/user-attachments/assets/bc1cca0e-66e2-4c3c-9343-f9beb4c53d86" />

- x, y 가 합쳐지는 노드 g = xy 가 있다고 해보자. backpropagation을 통해 파라미터를 학습해나가게 될 것
- 파라미터 x를 업데이트하기 위해서는 걔의 gradient를 알아야 그만큼 스텝을 더 밟아나갈 수 있음 - 마지막 항은 값 자체가 y가 됨
- y는 어떠한 activation function을 거쳐서 나오게 된 것임 - sigmoid 라고 했을 때, 만약 아주 작은 값이 들어갔다면 y = 0.0000001이 등장하게 됨
- 한 번 이렇게 되어버리면, 그 전과 그 전, 계속 전쪽으로 추정해 나갈때 다 0에 가까운 값들로 나오게 될 것
- 즉, 앞쪽 (최종 쪽에 가까운 쪽) 은 어느 정도 학습이 된 상태이지만, 그 전으로 가게 되면 0에 가까운 값들만 들어오게 되므로 스텝을 밟아나갈 수가 없음 = `gradient vanishing`


<img width="739" height="267" alt="Image" src="https://github.com/user-attachments/assets/a5da45a2-d3f7-434f-a92a-f1b6f921f811" />

- 레이어를 쌓을수록 시그모이드가 곱해지고 곱해지는 효과가 남
- 시그모이드를 곱하면 곱할수록 거의 직선에 가까운 형태가 됨 - 어느 값이 들어가도 0에 가까워짐 - 학습하기가 어려운 상태가 됨
- 모델이 deep해질수록 학습이 되지 않는다는 문제

<br />      

## Ans : ReLU Activation

<img width="679" height="338" alt="Image" src="https://github.com/user-attachments/assets/80a1bddb-45fe-439d-8238-7bfa85be0c1d" />

- ReLU 등 아주 다양한 activation function이 나오게 됨
- sigmoid와 다른 함수들의 성능 차이가 엄청나게 남 - 딥하게 쌓을수록 더 잘됨 !

<br /><br />   

## Others

### Xavier Initialization

<img width="689" height="373" alt="Image" src="https://github.com/user-attachments/assets/e048bdb6-9f1f-43f3-9263-4c1e58a3fb70" />

- sigmoid를 사용할 때 가중치를 초기화하는 방법
- standard Gaussian Dist에서 랜덤하게 뽑아서 initialization을 했다고 생각하자
    - 작은 상수를 곱해서 initialization 값을 준 경우 : training을 하면 할수록 더 작은 값으로 가게 될 것이고, 점점 0에 가까운 상태가 될 것임
    - 큰 상수를 곱해서 Initialization 값을 준 경우 : training을 하면 할수록 절댓값이 더 큰 값으로 가게 될 것이고, 점점 -1 혹은 1에 가까운 상태가 될 것임
- 어떠한 수학적 방법을 사용하면, 이것들이 아무리 많은 activation을 거치더라도 Gaussian 모양을 잘 유지하면서 전달된다

### Batch Normalization

<img width="669" height="278" alt="Image" src="https://github.com/user-attachments/assets/f4c361e4-87ad-48c0-a58c-d5f4710cc755" />

---

# [Standalone DL] #14 Lecture - Hyperparameter Tuning 
## Option for Tuning

### Model Related

**Number of hidden layer** 

- flexible : 1~10 layer for MLP. 10개 넘어가면 의미가 없는 듯함.
- Recent CNN의 경우 더 많은 레이어를 가짐
    - ResNet의 경우 레이어가 152개임.
    - 그렇다면 Vanishing Gradient가 발생하지는 않나?
    
    → 그 모델의 또 해결 방안이 있음 ! 
    

**Number of hidden unit**

- 10개 ~ 2048개 정도 시도해볼 수 있음

**Activation Function**

- Sigmoid, tanh, ReLU, …

(현재 MLP 구조를 보고 있기 때문에 몇 없지만, CNN으로만 가도 더 많은 요소들이 추가가 될 것임)

<br />   

### Optimization Related

**Type of Optimizer**

- GD : 데이터셋 전체를 학습시키는 방식
- SGD : 데이터셋을 chunking하여 학습시키는 방식
- RMSProp
- ADAM : 가장 많이 사용하는 방식
- AdaDelta
- 최신에 나온 것이라고 무조건 잘 하는 것은 아님. 간단한 태스크에 대해서는 SGD가 가장 잘 함

**learning rate** 

- 1e-5 ~ 1e-1 범위 내에서 진행

**L2 Coef**

- 1e-5 ~ 1e5 범위 내에서 진행

**Dropout** 

- 0.1 ~ 0.5 범위 내에서 진행
- 모델이 너무 복잡한데 데이터가 부족하다면 0.7정도 까지 쓰는 듯함

**Batch Size** 

- 128, 256, 512 정도의 값을 사용
- 우선 batch size를 최대한 키움 ( = GPU에 올리는 데이터가 많아진다는 것) → `out of memory error`가 뜰 때가 있음
- 그것이 발생하는 언저리까지 올리기 (그래야 빨리 끝나기 때문)
- overfitting이 되었다면 batch size를 줄여본다 (이것이 왜 overfitting의 문제를 해결해주는가? 에 대한 의문이 생길 수 있는데, 이는 다음에 ..)

**Epoch**

- overfitting을 확인해야 함
- val_loss를 train_loss와 함께 트래킹해야함. n_epoch에서 val_loss가 줄어들지 않는다면, 2-3번의 epoch 이후 멈춰주는 것이 좋음

<br/>

## 4 Ways to Tune Exp

### Method
- 그리드 서치를 해서, hyperparameter 변화에 대한 전반적인 경향성 확인하기
- 랜덤 서치하며 좋은 조합 선택
- 지속적으로 하이퍼 파라미터 튜닝
- 파악이 되면 Bayesian optimization - 범위를 정하고 프로그램을 돌려서 진행 !

### Caution

- test accuracy 보다, val accuracy 를 확인하면서 학습을 진행해야 함 !