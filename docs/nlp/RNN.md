---
layout: default
title: "RNN"
subtitle: 이전 출력을 현재 입력으로 사용하는 순환 구조, RNN 
permalink: /nlp/rnn/
use_math : true
nav_order: 1
parent: nlp
---

# [RNN] Recurrent neural network based language model (2010)

## Intro

sentence를 구성하고 단락을 구성하기 위해서는 기본적으로 단어들의 sequence가 중요하다. Statistics Language Model은 주어진 맥락에서 다음 단어를 예측하기 위한 것이다. 이를 위해서 이런 저런 시도들이 있었다. 

1. N-gram : 특정 순서로 인접한 n개의 기호의 시퀀스이다. 
2. Cache models and class-based models : 긴 맥락의 정보를 묘사하고, 그 단어들 사이에 parameter을 공유해서 나름 개선이 된다. 그치만 long-term dependencies를 stochastic GD에 의존해서 학습을 진행하기에는 한계가 있다. 



## Model Description

### 구조

simple RNN이라고 불리는 모델을 만들었다. 

구성 요소는 input layer x, hidden layer s, output layer y 이다. input은 모두 시간과 관련된 함수로 표현된다 (문맥은 시계열 데이터이기 때문.) input vector x 는, 현재 단어를 의미하는 vector w와 이전 레이어의 output을 concatenate해서 만들어진다 (그냥 .. 더한다는 것). 

\[
x(t) = w(t) + s(t - 1) \tag{1}
\]

hidden layer에서는 다음과 같은 계산이 이루어진다. 

\[
s_j(t) = f\left(\sum_i x_i(t) u_{ji}\right) \tag{2}
\]

여기서 f는 시그모이드 함수이다. hidden layer에서의 출력은 보통 확률값을 나타내는데, 시그모이드를 통해 미세한 변화도 확률값으로 변환해서 사용할 수 있다. 

그리고 최종적으로 y에서는 이런 계산이 이루어진다. 

\[
y_k(t) = g\left(\sum_j s_j(t) v_{kj}\right) \tag{3}
\]

여기서 g는 소프트맥스 함수이다. softmax를 통해 output의 확률분포를 얻을 수 있다. 



### 학습하기

학습할 데이터가 많을 경우 초기값은 중요하지 않다. 일반적으로 벡터 x의 크기는 단어 V의 크기 (30,000 ~ 200,000) 와 context layer의 크기를 더한 값이다. hidden layer의 크기는 보통 30~500 unit이다 (훈련 데이터의 양을 반영해야 함). 

학습은 standard back-propagation과 SGD로 진행된다. 

- 초기 learning rate는 0.1로 시작하는데, 매 epoch마다 훈련된 모델에 대해 validation data로 테스트를 진행한다. validation data의 log-likelihood가 증가할 경우 새로운 epoch에서 학습을 계속한다 (log-likelihood가 증가한다 = 더 좋은 성능을 낼 수 있도록 학습될 여지가 있다).
- 어떤 중요한 변화가 관찰되지 않기 시작하면, learning rate를 매 epoch에서 절반으로 낮춘다.

```
learning rate가 낮다는 게 무슨 뜻?

모델이 학습을 진행할 때에는 loss function이 가장 낮은 지점을 찾아간다는 것이다. 그 가장 낮은 지점으로 이동하기 위해 learning rate에 따라 이동하며 찾아나간다고 생각하면 되는데, learning rate가 클 경우 한 번에 이동하는 거리가 크다, 즉 보폭이 크다는 것이다. learning rate가 작을수록 점점 더 미세하게 찾아나간다는 의미이다. 
```

- 그리고 더이상의 중요한 변화가 관찰되지 않으면 학습이 끝난다. 보통 10-20 epoch에서 달성된다.



### 결과와 업데이트

아주 큰 사이즈의 hidden layer을 사용하더라도 overtrain이 발생하지 않는다고 한다. 

output layer은 이전 단어와 그 때까지의 맥락을 토대로 다음 단어의 확률 분포를 나타내는 형식이며, error vector은 엔트로피 기반이고 weight는 back-propagation으로 계산된다. 

- error vector : desired(t) - y(t)

desired(t)는 1-of-N coding을 기반으로 만들어진 벡터이다. 

```
1-of-N coding이란?

데이터셋을 단어별로 쪼개서 확률처럼 만들어둔 벡터이다. 정답 단어의 확률을 1로, 나머지는 0으로 설정하여 network가 예측한 확률 분포 (y(t))와 정답 분포 간의 차이를 효과적으로 계산할 수 있게 한다. 
```

일반적인 Statistics Language Model에서 훈련 단계와 검증 단계에서는, 테스트 데이터로 확인하는 동안 모델이 업데이트되지 않는다. 그러나 이 모델은 `dynamic model`로, 검증 단계에서도 모델이 새로운 데이터를 처리하면서 가중치가 업데이트된다. 

dynamic model이 필요한 이유는 다음과 같다. 

1. 새로운 데이터의 처리 : 테스트 데이터에서 새로운 단어나 정보가 반복적으로 등장할 수 있는데, 이때 학습을 통해 파라미터를 업뎃하지 않으면 해당 단어를 예측할 확률이 매우 낮게 나올 것 .. 그러면 모델 성능을 과소평가하게 될 것이다. 
2. long-term memory : 문맥에 대한 정보를 network의 가중치에 저장하도록 유도해서, 검증 단계에서도 모델이 점진적으로 학습하도록 만든다.  

즉, dynamic model은 새로운 도메인에 자동적으로 적응할 수 있을 것이다. 

dynamic model에서는 learning rate =0.1로 픽스된 값을 사용한다. 훈련 시 모든 데이터는 각각 epoch에 제공되지만, dynamic model은 검증 단계에서 한 번만 업데이트된다 (optimal한 솔루션은 당연히 아니겠지만, 이것만으로도 static model에 비해 perplexity reduction을 크게 얻을 수 있다) 

```
perplexity reduction이란? 

모델이 uncertainty를 얼마나 잘 줄이는지를 평가하는 것이다. 즉, perplexity 값이 작을수록 좋은 것. Reduction이 목표가 되는 것. 
```



### 최적화

성능 향상을 위해, 훈련 text들에서 특정한 임계값을 설정하고 그보다 더 적은 횟수로 발생하는 단어들을 rare token으로 합친다. 그 단어의 확률은 다음과 같이 계산된다. (짜잘한 것들은 한번에 묶어서 따로 처리하는 느낌). 

\[
P(w_{i}(t+1)|w(t), s(t-1)) =
\begin{cases}
y_{\text{rare}}(t) C_{\text{rare}} & \text{if } w_i(t + 1) \text{ is rare} \\
y_i(t) & \text{otherwise}
\end{cases} \tag{7}
\]

특정 임계값보다 적은 확률로 발생하는 애들은 그냥 똑같이 취급된다는 것이 특징이다.
