---
layout: default
title: "04 Seq2Seq"
permalink: /nlp/seq2seq/
subtitle: 서로 다른 길이의 입력, 출력을 처리하는 Seq2Seq
use_math : true
parent: nlp
---

# [Seq2Seq] Sequence to Sequence Learning with Neural Networks

## Intro

DNN은 1) speech recognition과 2) visual object recognition을 잘 수행하는 아주 강력한 머신러닝 모델이다. 그러나, 인코더 벡터의 차원이 제한되어 있다는 점이 가장 큰 문제점이었다. 왜냐면 미리 그 벡터의 차원을 알 수 없는 (lengths are not known a-priori) sequence들을 잘 표현할 줄 아는 것이 중요하기 때문이다. 

그래서, Seq2Seq에서는 두 개의 LSTM을 이용한다. 

- input을 읽는 LSTM : 한 번에 하나의 timestep를 읽어나가며 large fixed-dimensional vector를 만든다
- output을 처리하는 LSTM : 그 large vector에서 output sequence를 추출한다

특히. 두 번째 LSTM은 `RNN language model`이다 (input sequence 조건에 따른다는 점만 제외하고). `RNN language model`은 주어진 이전 단어들의 시퀀스를 기반으로 다음 단어를 예측한다. 예를 들어, i am 이 있으면 다음에 going이 나올 확률이 얼마나 되는지를 분포를 통해 예측하는 것이다. 방식이 유사하긴 하지만 참조하는 시퀀스가 '주어진 이전 단어들'이 아닌, 'input LSTM에서 읽어낸 large vector'이라는 점이 다르다. 

## The Model

### LSTM을 사용하는 이유

RNN은 sequence를 처리할 때, FFNN의 가장 자연스러운 generalization이다. 일반적으로 다음 수식을 통해 계산이 된다. 

\[
h_t = \sigma(W_{hx}x_t + W_{hh}h_{t-1}), 
y_t = W_{yh}h_t
\]

RNN은 input과 output의 길이가 미리 알려져 있을 때, sequence를 쉽게 매핑할 수 있다. 그러나, input과 output이 서로 다른 길이를 가지고 있거나, 일대일 매칭이 아닌 경우 (non-monotonic relationships) RNN을 적용하기 어렵다. 

일반적인 sequence learning은 먼저 한 RNN으로 input sequence를 fixed-size vector로 만들고 그 다음 다른 RNN을 통해 target sequence로 매핑시키는 것이다. RNN은 전체 맥락을 모두 제공받기 때문에 논리 상으로는 가능하긴 한데, long term dependencies 이슈로 RNN을 훈련하는 것 자체가 어려울 수 있다. 그러나 LSTM은 long-range도 잘 하므로 LSTM 두 개로 모델을 구성한 것이다. 

### LSTM의 목적

LSTM의 목표는 조건부확률 $p(y_1, \ldots, y_{T'}) | (x_1, \ldots, x_T)$를 추정하는 것이다. ($x$들은 입력 시퀀스, $y$들은 그에 대응하는 출력 시퀀스, 서로 길이는 다를 수 있다는 것을 가정) 

이 조건부확률을 구하기 위해 다음과 같은 과정을 거친다. 

1. input sequence의 fixed dimensional representation를 구한다 - LSTM의 마지막 hidden state에 있을 것임. hidden state에 맥락이 차근차근 저장이 되어 오고 있었을 것이므로. 
2. LSTM-LM 형식을 사용해서 $y_1, \ldots, y_{T'}$의 확률을 구한다 

LSTM-LM 형식은 다음과 같다. 

\[
p(y_1, \ldots, y_{T'} | x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t | v, y_1, \ldots, y_{t-1}) \tag{1}
\]

Seq2Seq의 특이점은 바로 <EOS> 토큰이다. 

각 문장은 끝에 special end-of-sentence symbol인 <EOS>를 달게 된다. 이를 통해 모델이 sequence가 언제 끝났는지 알 수 있고, 다양한 길이의 sequence를 처리할 수 있게 된다. 

예를 들어 input 쪽 LSTM은 "A", "B", "C", <EOS>를 처리하고, output 쪽 LSTM은 이를 기반으로 "W'", "X", "Y", "Z", <EOS>의 확률 분포를 계산한다. 이 출력 단어들의 확률 분포를 기반으로 softmax를 취해 각 단어에 대한 확률을 계산한다. 

### Seq2Seq의 differ point

Seq2Seq에는 세 가지 중요한 포인트가 있다. 

1. **두 개의 서로 다른 LSTM을 사용한다.** 

하나는 input sequence를 위한 것이고, 다른 하나는 output sequence를 위한 것이다. 이렇게 하면 모델 파라미터도 수용 가능한 선 정도로 늘어나고, 두 개의 언어 세트를 동시에 훈련하기에도 자연스럽고 좋다. 

```
모델 파라미터랑 무슨 관련이 있어? 

만약 동일한 LSTM을 사용한다면, 이 LSTM은 입/출력을 모두 처리할 수 있도록 만들어져야 한다. 이말인즉슨, LSTM이 해야 되는 일이 더 많아지니 더 많은 파라미터를 학습해야 한다는 뜻이다. 
서로 다른 LSTM을 사용할 경우, 각 LSTM은 자신이 담당하는 작업 (입력 or 출력) 만 진행하면 되니 적절한 양의 파라미터로 승부할 수 있다. 
```

```
두 개의 언어 세트를 동시에 훈련한다는 게 뭐야? 

LSTM이 두개니까, 각 LSTM이 각 언어를 전담해서 학습할 수 있다. 추가적으로, 확장성도 가질 수 있을 것이다. 같은 인코더를 두고, 여러 디코더를 사용하여 서로 다른 출력 언어를 처리할 수도 있다. 
```

1. **4 layers를 가진 LSTM을 사용한다.** 

deep LSTM이 shallow LSTM보다 훨씬 성능이 좋다는 것을 발견하였다. 

1. **input sentence를 뒤집을 때 성능이 더 좋다.**

예를 들어 문장 a, b, c를  ****α, β, γ 와 매핑하는 것보다 c, b, a를  α, β, γ와 매핑하는 것이 더 좋다는 것이다 ( α, β, γ는 각각 a, b, c의 번역이다). 이렇게 했을 때  α의 뜻이 a에 더 가깝고, β의 뜻이 b에 더 가깝고, γ의 뜻이 c에 더 가깝다는 것이다. 또한, SGD가 더 잘 작동한다. 

```
왜 그런거지? 

Seq2Seq 모델은 input sequence를 읽은 후 -> 고정된 크기의 large vector로 요약되어 -> 디코더가 이를 기반으로 output sequence를 만들어낸다. 
즉, input sentence의 앞부분은 초반에 처리가 될 것이고, 뒤의 문장들이 다 처리가 된 이후에 디코더가 이를 활용할 것이다. 결국 문장이 길어지게 되면 그 문맥이 희석되어 input-output dependency가 약해질 수 있는 것 !! 

SGD에서 생각을 해보면, 순서를 뒤집어서 위와 같이 더 잘 동작하게 될 때 alignment가 강화되어 vanishing gradient problem (기울기 소실 문제) 가 완화될 것이다. 특히, 긴 시퀀스를 처리할 때 더더욱 그럴 것 !! loss function이 더 안정적으로 수렴하게 되어 훨 효율적이다. 
```
