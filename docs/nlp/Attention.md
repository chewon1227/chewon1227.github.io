---
layout: default
title: "[Attention] Neural Machine Translation by Jointly Learning to Align and Translate (2015)"
permalink: /nlp/attention/
subtitle: 입력 각 부분이 출력에 얼마나 중요한지 계산하여 '집중'할 수 있도록 하는 Attention
use_math: true
parent: nlp
---

## 1. Intro

**Neural Machine Translation**은 기계번역에서 새롭게 등장하는 접근이다.

기존에 있었던 Traditional Phrase-based translation system의 경우 sub component들이 있고 각각 tune되어야 했지만, 이 neural machine translation은 1개의 거대한 신경망 구조이다 (확실히 학습 시 효율적일 것)

일반적으로 기계번역 모델이라고 하면 encoder-decoder 형식이고, 각 언어가 encoder, decoder을 각각 차지한다. encoder이 기존 문장을 fixed-length 벡터로 읽어들인 후, decoder이 그에 대한 번역을 도출한다. 여기서 fixed-length 벡터로 읽어들인다는 점이, 긴 문장을 대하기 어렵다는 점에서 한계점이다. 특히 훈련 코퍼스에서 긴 문장이 있을 시 제대로 학습이 안될 것이다.

그래서 align과 translate를 동시에 학습하는 encoder-decoder model을 제시한다.
	•	translation으로 단어를 생성해서 제시함
	•	가장 관련된 정보가 집중되어 있는 source sentence에서의 set of positions 를 탐색함
	•	source position과 이전에 생성된 모든 target words 를 기반으로 target word를 예측함

특히 가장 주목할만한 특징은 fixed-length 구조가 아니라는 것이다. 즉, 그 길이에 맞춰서 자를 필요가 없어졌으니 긴 문장에 더욱 강력한 성능을 뽐낼 것이다.

---

## 2. Background

확률적 관점에서 보면 기계번역은, source sentence $x$에 대해서 target sentence의 conditional probability $y$를 최대화하는 것이다. 이를 위해 신경망 구조를 활용하는 것이 대두되었고, 특히 RNN을 두 개 사용하는 방식으로 구현된다.
	•	한 RNN은 다양한 길이의 source sentence를 fixed-length 벡터로 encode하는 데에 사용
	•	나머지 RNN은 다시 다양한 길이의 target sentence로 decode하는 데에 사용

즉, 정리하면 variable-length → fixed length → variable-length 로 가는 것이다.

### 2-1. RNN Encoder-Decoder

encoder이 input sentence를 읽는다 - $x = (x_1, \ldots, x_{T_x})$ 를 $c^2$로 변환해서! hidden layer은 일반적으로 $h_t = f(x_t, h_{t-1})$ 와 같은 형태로 구성된다. 그리고

$$
c = q({h_1, \ldots, h_{T_x}})
$$

를 통해 context vector $c$를 계산한다.

decoder은 $c$와 이전에 만들어진 단어들 ${y_1, \ldots, y_{t’-1}}$ 을 고려하여 다음 단어인 $y$를 예측한다.

$$
p(y) = \prod_{t=1}^{T} p(y_t \mid {y_1, \ldots, y_{t-1}}, c),
$$

$$
p(y_t \mid {y_1, \ldots, y_{t-1}}, c) = g(y_{t-1}, s_t, c),
$$

여기서 함수 $g$는 확률을 끌어내기 위한 비선형 함수가 될 것이다.

---

## 3. Learning to Align and Translate

align과 translate를 어떻게 동시에 학습할 수 있을까? 이를 위해 새로운 구조를 도입한다.
	•	encoder : bidirectional RNN
	•	decoder : source sentence를 탐색하는 구조를 모방


### 3-1. Decoder

기존 RNN Encoder-Decoder 모델에서는

$$
p(y) = \prod_{t=1}^{T} p(y_t \mid {y_1, \ldots, y_{t-1}}, c),
$$

로 정의했던 $p(y)$를 이제는

$$
p(y_i \mid y_1, \ldots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i),
$$

로 정의한다. 가장 큰 차이점은, 각 단어 $y_i$마다 독립적인 context vector $c_i$를 사용한다는 것이다.

각각의 $c_i$는 encoder이 생성한 annotation의 순서에 따라 계산된다. 이때 annotation은 전체 input 순서에 대한 정보를 가지고 있으며 특히, $i$번째 단어 주변에 강하게 초점이 맞추어져 있다. $c_i$는 다음과 같이 계산된다.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

여기서 가중합 $\alpha_{ij}$는 다음과 같이 계산된다.

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$\alpha$는 translation 단어를 생성할 때 얼마나 각 context에 attention할 건지를 나타낸다. 이는 alignment model이라고 할 수 있는데 input의 $j$ 번째 단어와 output의 $i$ 번째 단어가 얼마나 매칭되는지를 점수매긴다.

즉, decoder는 source sentence에서 어느 위치의 단어에 더 attention을 줄지를 결정할 수 있다.


### 3-2. Encoder : Bidirectional RNN for Annotating Sequence

encoder에서는 기존과는 달리 양방향으로 읽어나간다.
	•	Forward RNN의 경우 순서대로 읽어나가며 forward hidden state를 생성한다
	•	Backward RNN의 경우 거꾸로 읽어나가며 backward hidden state를 생성한다

---

## 4. Qualitative Analysis

### 4-1. Alignment

weight $\alpha_{ij}$ 를 통해 soft-alignment를 찾을 수 있게 되었다.


### 4-2. Long Sentences

기존 RNN은 문장이 길 경우 후반부 번역이 흐려졌지만, 새 모델은 잘 해낸다.

---

## 5. Model Architecture

### 5-1. Recurrent Neural Network - Gated Hidden Unit

activation function $f$ 자리에는 gated hidden unit이 들어간다. LSTM과 유사한 구조로 long-term dependency를 잘 학습한다.

$$
s_i = (1 - z_i) \odot s_{i-1} + z_i \odot \tilde{s_i},
$$

$$
z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i), \quad
r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)
$$


### 5-2. Alignment Model

$$
a(s_{i-1}, h_j) = v^T \tanh(W_a s_{i-1} + U_a h_j)
$$


### 5-3. Encoder

입력 $x = (x_1, \ldots, x_{T_x}), \quad x_i \in \mathbb{R}^{K_x}$
출력 $y = (y_1, \ldots, y_{T_y}), \quad y_i \in \mathbb{R}^{K_y}$

Forward state (bidirectional RNN)

$$
\overrightarrow{h_i} = (1 - \overrightarrow{z_i}) \odot \overrightarrow{h_{i-1}} + \overrightarrow{z_i} \odot \tilde{\overrightarrow{h_i}},
$$

$$
\tilde{\overrightarrow{h_i}} = \tanh(\overrightarrow{W} E x_i + \overrightarrow{U} [\overrightarrow{r_i} \odot \overrightarrow{h_{i-1}}]),
$$

$$
\overrightarrow{z_i} = \sigma(\overrightarrow{W_z} E x_i + \overrightarrow{U_z} \overrightarrow{h_{i-1}}),
$$

$$
\overrightarrow{r_i} = \sigma(\overrightarrow{W_r} E x_i + \overrightarrow{U_r} \overrightarrow{h_{i-1}})
$$

### 5-4. Decoder

$$
\tilde{s_i} = \tanh(W E y_{i-1} + U [r_i \odot s_{i-1}] + C c_i),
$$

$$
z_i = \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i),
\quad
r_i = \sigma(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
$$

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$


## 6. 정리

<div class="callout">

	•	Attention이란
→ hidden layer 정보를 기반으로, 다음 단어 예측 시 중요한 정보를 집중하는 메커니즘
	•	수행 과정
→ alignment 계산 → attention weight 계산 → context vector 생성 → decoder 업데이트

</div>