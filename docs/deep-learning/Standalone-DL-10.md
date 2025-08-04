---
layout: default
title: "10 Advanced Optimizer"
parent: deep-learning
permalink: /dl/sa/standalone-10/
use_math : true
---

# [Standalone DL] 10 Lecture - #17 Advanced Optimizer than SGD 

## Problem

overfitting을 해결하기 위해 L2 Regularization, Dropout이라는 해결책 제시 

- L2 Regularization : Loss = MSE + ( ) → 파라미터들이 너무 큰 값으로 explode하지 않도록 하는 방식
- Dropout : 모델 capacity를 인위적으로 감소시키고, 하나의 복잡한 모델을 학습하기보다는 여러 가지의 단순화된 모델을 학습시키는 방식

Gradient Vanishing을 해결하기 위해 ReLU Activation Function 제시 

- Gradient가 0인 부분을 거치게 되면 그 뒤 Gradient는 다 0이 되어서 vanishing 문제 발생
- Sigmoid 대신 ReLU Function을 제시 - Gradient가 양 끝단에서 0으로 수렴하지 않고 값을 잘 유지하며 거쳐갈 수 있게 됨

MLP를 더 잘 훈련시키기 위한 Xavier Initialization와 Batch Normalization 제시 

- Xavier Initialization : 너무 큰 값으로 시작하면 수렴하지 않고 불안정하게 무너지게 됨. 수학적으로 특정한 형태로 Initialization을 하면 잘 유지가 된다
- Batch Normalization : ReLU 함수에서 0보다 큰 쪽에만 데이터가 몰려 있으면 아무리 여러 개의 레이어를 쌓아도 결국 자기 자신만 계속 배출하므로 한 개의 레이어를 쌓은 것과 다름이 없다는 문제를 해결하기 위함 - 적절한 Normalization을 통해 activation이 제 역할을 할 수 있도록 함

<br />   

## Batch / Stochastic Gradient Descent

### Problem

$$
\theta = \theta - \eta \nabla J(\theta)
$$

$\theta$ : parameter set of the model 

$\eta$ : learning rate 

$\nabla J(\theta) $ : Loss function 

- 이미지같이 데이터가 큰데 수가 많은 경우, RAM의 한계 때문에 너무 적은 training set을 보게 될 것
- 한 스텝을 밟는데 모든 training set을 다 봐야 한다는 문제점이 있음

### Stochastic Gradient Descent

<img width="647" height="386" alt="Image" src="https://github.com/user-attachments/assets/efeaf77a-83a4-4da2-89be-a0d648c7804b" />

전체 training dataset에서 small chunk (`mini-batch`) 의 gradient를 계산하는 방식이다. 

- deterministic (전체를 다 확실하게 보는 방식) 하지 않고, **stochastic** (확률적인 방식) 하다. 즉, mini-batch depended하다.
- batch GD보다는 확실히 빠르지만, 비슷하게 **convergence**하기 때문에 사용하는 방식이다.
    - Batch를 올리기만 하면 사이즈에 상관 없이 병렬적으로 계산함 (시간이 거의 유사함)
    - batch GD의 경우 전체 training set을 램에 올릴 수 없기 때문에 나눠서 넣게 되면 - 시간이 오래걸릴 수 밖에 없음
    - Batch마다 스텝을 밟게 되면 계산 → step → 계산 → step → .. 이므로 batch size를 적절하게 키우거나 조절하면 빠른 속도를 낼 수 있음
- local minima 를 피할 수 있다
    - random으로 pick한 것을 계산하는 것이기 때문에 약간 피해갈 수 있음
    - (추가적인 알고리즘이 더해져야 완전히 피할 수 있음)

### Problem of Vanilla SGD

- local minima에 갇히게 된다
- 이를 해결하기 위한 방법 : `momentum`

<br />   

## Solving the Vanilla SGD Problem

### Momentum

$$
\theta = \theta - v_t \\
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)
$$

- 개념
    - 공을 굴리듯이 = 관성을 고려해서 = 운동량을 고려해서 local minima를 빠져나올 수 있을 것이라는 직관
    - $\gamma$가 없다면 계속 양수값 $v_{t-1}$이 더해지는 것 - $v_{t}$가 계속 커질 것. 이것을 방지하기 위해 전 값들이 적절하게 줄어들고 안정화될 수 있도록 함
    - 중요한 차이점dms, 이전까지는 직전의 gradient만 가지고 스텝을 밟아나갔음. 그러나 momentum의 경우 이전 모든 스텝들의 값들을 다 받아서 gradient를 밟아나가게 됨. **이전 방향성의 관성을 지속적으로 고려하는 것** !
- 해결할 수 있는 문제점
    - local minima에 stuck하는 문제
    - vanilla의 경우 방향을 random하게 잡기 때문에 급격하게 틀어버리는 경우가 많아서 비효율적인데, momentum의 경우 경로를 smooth하게, 즉 더 빠르게 minima로 갈 수 있다는 장점
- 생기는 문제점
    - global minima에 도달하더라도 관성값 때문에 계속 왔다갔다하게 됨

### Nesterov Accelerated Gradient (NAG)

<img width="691" height="235" alt="Image" src="https://github.com/user-attachments/assets/69788a3b-8c53-4f4f-aa52-89a831d472ed" />

$$
\theta = \theta - v_t \\
v_t = \gamma v_{t-1} + \eta \nabla_\theta J\left(\theta - \gamma v_{t-1}\right)
$$

- 개념
    - $\theta$ 자리에 $\theta - \gamma v_{t-1}\right)$가 들어감 - 원래 자리에서 momentum을 고려한 새로운 자리에서 gradient를 계산하겠다는 원리
    - global minima에 도착했을 때, 원래 momentum을 가지고 간다면 그 자리에서 튕겨나갈 것. 그러나 NAG를 사용하면 그 튕겨남의 수준이 낮아질 것 - **Fast Convergence** !

### Problem of SGD

지금까지의 GD가 가진 공통적인 문제점 : **Step Size is Equal for Every Parameter** 

- 각각의 파라미터마다 다른 크기의 step을 밟아야 하는 것 아닌가?
- learning rate를 각 파라미터에 맞춰 나가야 하는 것 아닌가?

**→ Adaptive Gradient (Adagrad)** 

### Adagard

$$
\theta_{t+1} = \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot \nabla_\theta J(\theta_t) \\
G_t = G_{t-1} + \left( \nabla_\theta J(\theta_t) \right)^2
$$

- 개념
    - $G_{t}$ : k개의 파라미터가 있다고 생각했을 때, 각각의 값에는 각 파라미터에 대응되는 Gradient의 제곱값이 지속적으로 더해지게 됨 (즉, 지금까지의 모든 변화 과정을 반영하게 됨)
    - 원래는 element-wise하게 곱해서, Gradient 값을 동일한 learning rate $\eta$에 곱하자는 아이디어 ⇒ **각각의 learning rate를 곱해주자**는 아이디어로 ! 이때 learning rate는, 지금까지 변해온 만큼과 반비례하는 값이 될 것임
    - 즉, learning rate를 파라미터마다 자동으로 조절하자는 개념. 자주 업데이트되는 파라미터는 learning rate를 낮추고 (global minima에 가까울 확률이 높으므로), 드물게 업데이트되는 파라미터는 learning rate를 높이자.
- 해결할 수 있는 문제점
    - 각 파라미터에 맞게 learning rate를 조절할 수 있게 됨
- 생기는 문제
    - G가 계속 커질 것임 (양수값을 지속적으로 더해나가기 때문)
    - 그러면 step size는 0으로 decay하게 될 것임 ..

### RMSProp

$$
\theta_{t+1} = \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot \nabla_\theta J(\theta_t) \\
G_t = \gamma G_{t-1} + (1 - \gamma) \left( \nabla_\theta J(\theta_t) \right)^2
$$

- 개념
    - $G_t$에 decay constant인 $\gamma$, $1-\gamma$가 붙음
    - 지금까지 더해온 값과 새로 더할 값에 0과 1 사이의 어떤 값을 곱해서 더해주면, $G_t$가 무한으로 발산하지 않고 **지속적으로 스텝을 밟아나갈 수 있게** 됨
- 해결할 수 있는 문제점
    - $G_t$가 무한대로 발산하여서 업데이트가 안될 수도 있는 문제를 해결함

### AdaDelta

$$
\theta_{t+1} = \theta_t - \Delta_\theta\\
\Delta_\theta = \frac{\sqrt{s + \epsilon}}{\sqrt{G + \epsilon}} \cdot \nabla_\theta J(\theta_t) \\ 
s_{t+1} = \gamma s_t + (1 - \gamma) \Delta_\theta \\ 
G_{t+1} = \gamma G_t + (1 - \gamma) \left( \nabla_\theta J(\theta_t) \right)^2
$$

- 개념
    - $\theta_{t+1} = \theta_t - \Delta_\theta$ 에서 $\theta_{t+1}$ 의 물리량을 $\u$라고 생각해보자 ! $\Delta_\theta$를 원래 방식대로  $\eta \nabla J(\theta$ 로 계산할 경우 - $\theta$에 대한 loss의 평균값이므로 이 항만 물리량이 $(\u)^(-1)$이 됨 ⇒ 이 부분을 문제삼았음.
    - 단위를 맞추기 위해 새로운 $\Delta_\theta$를 정의함
- 해결할 수 있는 문제점
    - initial learning rate를 지정해주지 않아도 알아서 잘 돌아가게 됨
- 실제로 잘 작동하지는 않음

### Adaptive Moment Estimation (ADAM)

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

( 여기서 $g_t$는 $\nabla_\theta J(\theta_t)$의 약자임 ) 

- 개념
    - $m_t$ : momentum의 $ v_t $ 와 비슷함. momentum을 나타내는 값
    - $v_t$ : gradient의 제곱을 지속적으로 더하고 있음. adaptive learning rate를 적용하기 위한 값
    - 이 두 값을 모두 고려함으로써, `first-moment gradient`와 `second-moment gradient`를 모두 고려하겠다는 것. loss surface를 이차함수로 근사하겠다 - convex function에 대해서는 더 빠르게 minimum으로 찾아갈 수 있기 때문
- practical use
    - 초기 설정 시 $\beta_1$ = 0.9, $\beta_2$ = 0.9999, $m_{t-1}$ = $v_{t-1}$ = 0 (이전 값이 없으므로)
    - 즉, initial step 시 $m_t$의 값은 엄청 큰데 $v_t$의 값은 엄청 작아서, 엄청 큰 step을 밟게 되는 문제가 있음. 그래서 $ \hat{m}_t ,  \hat{v}_t $ 를 통해 적절한 크기의 step을 밟아나갈 수 있도록 해줌

### How to use Advanced Optimizers in Pytorch?

그냥 코드 한 줄 쓰면 된다 😅

`optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`

- SGD 를 Adam 등등의 optimizer name으로 바꿔주면 된다