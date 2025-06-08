---
layout: post
title: "[Attention] Neural Machine Translation by Jointly Learning to Align and Translate (2015)"
subtitle: 입력 각 부분이 출력에 얼마나 중요한지 계산하여 '집중'할 수 있도록 하는 Attention
use_math : true
categories: Paper
---

## 1. Intro

Neural Machine Translation 은 기계번역에서 새롭게 등장하는 접근이다. 

기존에 있었던 Traditional Phrase-based translation system의 경우 sub component들이 있고 각각 tune되어야 했지만, 이 neural machine translation은 1개의 거대한 신경망 구조이다 (확실히 학습 시 효율적일 것) 

일반적으로 기계번역 모델이라고 하면 encoder-decoder 형식이고, 각 언어가 encoder, decoder을 각각 차지한다. encoder이 기존 문장을 fixed-length 벡터로 읽어들인 후, decoder이 그에 대한 번역을 도출한다. 여기서 **fixed-length 벡터로 읽어들인다는 점**이, 긴 문장을 대하기 어렵다는 점에서 한계점이다. 특히 훈련 코퍼스에서 긴 문장이 있을 시 제대로 학습이 안될 것이다. 

그래서 align과 translate를 동시에 학습하는 encoder-decoder model을 제시한다. 

- translation으로 단어를 생성해서 제시함
- 가장 관련된 정보가 집중되어 있는 `source sentence`에서의 `set of positions` 를 탐색함
- `source position`과 이전에 생성된 모든 target words 를 기반으로 target word를 예측함

특히 가장 주목할만한 특징은 fixed-length 구조가 아니라는 것이다. 즉, 그 길이에 맞춰서 자를 필요가 없어졌으니 긴 문장에 더욱 강력한 성능을 뽐낼 것이다. 

## 2. Background

확률적 관점에서 보면 기계번역은, source sentence x에 대해서 target sentence의 conditional probability y를 최대화하는 것이다. 이를 위해 신경망 구조를 활용하는 것이 대두되었고, 특히 RNN을 두개 사용하는 방식으로 구현된다. 

- 한 RNN은 다양한 길이의 source sentence를 fixed-length 벡터로 encode하는 데에 사용
- 나머지 RNN은 다시 다양한 길이의 target sentence로 decode하는 데에 사용

즉, 정리하면 `variable-length → fixed length → variable-length` 로 가는 것이다. 

### 2-1. RNN Encoder-Decoder

encoder이 input sentence를 읽는다 ($\x = (x_1, \ldots, x_{T_x})$ 를 c^2로 변환) . hidden layer은 일반적으로 $\h_t = f(x_t, h_{t-1})$ 와 같은 형태로 구성된다. 그리고 

\[
c = q({h_1, \ldots, h_{T_x}})
\]

를 통해 context vector c를 계산한다. 

decoder은 c와 이전에 만들어진 단어들 \( \{y_1, \ldots, y_{t'-1}\} \) 을 고려하여 다음 단어인 y를 예측한다. 

\[
p(y) = \prod_{t=1}^{T} p(y_t | \{y_1, \ldots, y_{t-1}\}, c),
\]

\[
p(y_t | \{y_1, \ldots, y_{t-1}\}, c) = g(y_{t-1}, s_t, c),
\]

여기서 함수 g는 확률을 끌어내기 위한 비선형함수가 될 것읻다. 

## 3. Learning to Align and Translate

align과 translate를 어떻게 동시에 학습할 수 있을까? 이를 위해 새로운 구조를 도입한다. 

- encoder : bidrectional RNN
- decoder : source sentence를 탐색하는 구조를 모방

### 3-1. Decoder

기존 RNN Encoder-Decoder 모델에서는 

\[
p(y) = \prod_{t=1}^{T} p(y_t | \{y_1, \ldots, y_{t-1}\}, c),
\]

로 정의했던 p(y)를 이제는

\[
p(y_i | y_1, \ldots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i),
\]

로 정의한다. 가장 큰 차이점은, 각 단어 $\y_i$마다 독립적인 context vector $\c_i$를 사용한다는 것이다. 

각각의 c_i는 encoder이 생성한 `annotation`의 순서에 따라 계산된다. 이때 `annotation`은 전체 input 순서에 대한 정보를 가지고 있으며 특히, i번째 단어 주변에 강하게 초점이 맞추어져 있다. c_i는 다음과 같이 계산된다. 

\[
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j.
\]

여기서 가중합 $\alpha_{ij}$ 는 다음과 같이 계산된다. 

\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})},
\]

\alpha 는 translation 단어를 생성할 때 얼마나 각 context에 attention할 건지를 나타낸다고 생각하면 된다. 이는 alignment model이라고 할 수 있는데 이는 input의 j 번째 단어와 output의 i 번째 단어가 얼마나 매칭되는지를 점수매긴다. 즉, decoder의 input과 output에서 어떤 매칭이 연관성이 높은지를 계산한다고 생각하면 된다. 

즉, 이 구조를 통해 decoder은 source sentence에서 어느 위치의 어느 단어에 더 attention을 가해서 focus할 지 정할 수 있다. 

### 3-2. Encoder : Bidirectional RNN for Annotating Sequence

encoder에서는 기존과는 달리 양방향으로 읽어나간다. 

- Forward RNN의 경우 순서대로 읽어나가며 `forward hidden state` 를 생성한다
- Backward RNN의 경우 거꾸로 읽어나가며 `backward hidden state` 를 생성한다

## 4. Qualitative Analysis

### 4-1. Alignment

weight αij 를 통해 생성된 번역본의 단어들과 source sentence 사이의 soft-alignment를 찾을 수 있게 되었다. 또한 (여기서는 생략되었지만) 실험을 통해 source sentence 그 자체를 번역하는 작업에서, 각 단어가 어떤 위치에 속해있는지를 파악하는 것이 더 중요하다는 것을 알게 되었다. 

여기서 task로 수행하고 있는 번역 작업은 English - French 인데, 이건 논문 Figure 3에도 나와있듯이 보통 monotonic하다 (plot을 보면 연결되는 align이 보통 우하향하는 일직선 대각선으로 표시됨). 그치만 일부 부분은 조금 달라지기고 하고, 형용사와 명사같은 경우 두 언어에서 다른 순서로 정렬됨에도 잘 번역되는 것을 볼 수 있다. 

soft-alignment를 통해 단어를 다양하게 보다 보니 유동적으로 번역할 수 있다는 점도 드러났다. 

### 4-2. Long Sentences

RNN encoder-decoder 은 문장이 길 때, 중반까지는 잘 번역했지만 후반부부터 원본 문장의 의미를 벗어나게 되었다. 

그러나 새로 만든 모델의 경우 아주 잘 번역해낸다. 

## 5. Model Architecture

### 5-1. Recurrent Neural Network - **Gated Hidden Unit**

RNN 구조의 activation function f 자리에는 `gated hidden unit`이 들어간다. `gated hidden unit`은 이전에 사용되었던 전통적인 simple unit (ex. element-wise tanh) 과 다르고, LSTM에서 나왔던 것과 유사하다.  

- 모델이 long-term dependencies를 더 잘 학습할 수 있게 된다
- 도함수의 곱이 1에 가까운 computational path를 가지고 있기 때문에, vanishing effect에서 좀 벗어날 수 있다 (back-propagation을 좀 더 수월하게 할 수 있다)

그래서 그냥 `gated hidden unit` 을 쓰지 않고 LSTM 을 갖다가 써도 된다. 

n 개의 `gated hidden unit` 을 사용하는 RNN decoder에서의 새로운 상태 \( s_i \) 는 이렇게 계산된다. 

\[
s_i = f(s_{i-1}, y_{i-1}, c_i) = (1 - z_i) \odot s_{i-1} + z_i \odot \tilde{s_i},
z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i),
r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)
\]

- 수식
    - $\odot$  : element-wise 곱셈
    - $\sigma$ : logistic sigmoid function
    - $\tanh$ : 하이퍼볼릭 탄젠트 함수로 활성화 값 계산

- Gates
    - $z_i$ : update gates - 각각의 hidden unit이 이전 activation을 유지할 수 있도록 한다.
    - $r_i$ : reset gates - 이전 상태에서 어떤 정보가 얼마나 reset되어야 하는지를 결정한다.
- 가중치 행렬
    - W : 현재 입력 $\e(y_{i-1})$ 와 관련된 가중치 행렬이다. W를 통해 model의 hidden 공간에 투영한다.
    - U : $\s_{i-1}$ 와 관련된 가중치 행렬이다. U를 통해 이전 hidden 상태 정보를 현재로 전달해서 sequence의 맥락을 유지한다.
    - C : $\c_i$ 와 관련된 가중치 행렬이다.

decoder의 각 스텝에서 output 확률이 나올 것이고, softmax를 통해서 각 단어마다 output 확률에 대해 분포를 만든다. 

즉, 확률싸움이다. 학습 데이터셋에서 등장하는 단어들의 확률 분포를 통해서 그럴듯한 말을 만들어내는 것이다. 

### 5-2. Alignment Model

single-layer multilayer perceptron 을 사용한다. 

\[
a(s_{i-1}, h_j) = v^T \tanh(W_a s_{i-1} + U_a h_j)
\]

### 5-3. Encoder

가장 먼저 input인 source sentence은 `1-of-K coded word vector` 형태로 들어간다. output인 translated sentence도 `1-of-K coded word vector` . 

\[
x = (x_1, \ldots, x_{T_x}), x_i \in \mathbb{R}^{K_x} ,
y = (y_1, \ldots, y_{T_y}), y_i \in \mathbb{R}^{K_y}
\]

```
1-of-K coded word vector과 one-hot vector의 차이

사실상 같은 개념이다. 문맥에 따라 조금씩 다르게 읽힐 수 있다. 

1-of-K coded word vector 은 각 단어를 K차원의 벡터로 표현한다. 
해당 단어를 나타내는 위치에만 1이 있고 나머지는 0으로 채워진다.
단어 하나만 선택된다는 의미가 강함. 수학적으로 많이 사용되는 듯. 

One-hot vector은 단어, 범주형 데이터 등에서 특정 항목을 가리킬 때 쓴다. 
좀 더 단순화된 곳에서 쓰일 수 있을 것. 
```

**Forward State**

Bidirectional RNN에서 forward state는 다음과 같이 계산된다. 
w
\[

\overrightarrow{h_i} =
\begin{cases}
(1 - \overrightarrow{z_i}) \odot \overrightarrow{h_{i-1}} + \overrightarrow{z_i} \odot \tilde{\overrightarrow{h_i}}, & \text{if } i > 0 \\
0, & \text{if } i = 0
\end{cases}



\overrightarrow{h_i} = (1 - \overrightarrow{z_i}) \odot \overrightarrow{h_{i-1}} + \overrightarrow{z_i} \odot \tilde{\overrightarrow{h_i}},

\tilde{\overrightarrow{h_i}} = \tanh(\overrightarrow{W} E x_i + \overrightarrow{U} [\overrightarrow{r_i} \odot \overrightarrow{h_{i-1}}]),

\overrightarrow{z_i} = \sigma(\overrightarrow{W_z} E x_i + \overrightarrow{U_z} \overrightarrow{h_{i-1}}),

\overrightarrow{r_i} = \sigma(\overrightarrow{W_r} E x_i + \overrightarrow{U_r} \overrightarrow{h_{i-1}}),

\]

update gate $\overrightarrow{z_i}$ 를 통해 기존 정보와 새로운 정보를 적절히 혼합해서 과거 정보를 유지하면서도 새로운 정보를 입력받을 수 있다. 

- $(1 - \overrightarrow{z_i}) \circ \overrightarrow{h_{i-1}}$  를 통해 기존 정보를 얼마나 유지할 지 결정한다. (새로운 정보와의 상쇄를 통해 기존 정보를 부분적으로 유지)
- $\overrightarrow{z_i} \circ \widetilde{\overrightarrow{h_i}}$  를 통해 현재 입력과 문맥에 기반해서 새로운 정보를 얼마나 반영할 지 결정한다.

초기 입력 단어라면 기존 정보가 거의 없으므로  $\overrightarrow{z_i}$  가 크고 새로운 정보들이 거의 다 반영될 것이다. 만약 긴 문장에서 이미 context가 좀 반영된 시점이라면,  $\overrightarrow{z_i}$ 가 작고 기존 문맥이 더 많이 유지될 것이다. 

**Backward State**

backward state도 유사하게 계산된다. foward와 backward RNN은 단어 임베딩을 공유하긴 하지만 **가중치 행렬은 공유하지 않는다**. 

### 5-4. Decoder

본문에 거의 다 언급된 내용이라, 간단하게 짚고 넘어가자. 

\[
\tilde{s_i} = \tanh(W E y_{i-1} + U [r_i \circ s_{i-1}] + C c_i),
z_i = \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i),
r_i = \sigma(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
\]



context vector은 이렇게 계산된다. 

\[
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
\]

## 6. 정리

정리를 해보면 다음과 같다. 

- Attention이란
    - hidden layer의 정보를 저장해서 맥락 상 얼마나 저장하고 버릴 것인지 판단해서 다음 단어를 예측할 수 있도록 만드는 구조.
    - 어떤 부분을 내가 좀 더 주의해서 봐야 하는가? 에 답하는 과정.
- Attention이 수행되는 과정
    - alignment score 계산 : 현 decoder 상태와 annotation h 를 기반으로 ‘얼마나 중요한지’를 측정하는 alignment 계산 (FFNN 이용)
    - attention 가중치 계산
    - context vector 계산
    - decoder 업데이트
    - 출력 단어 생성
