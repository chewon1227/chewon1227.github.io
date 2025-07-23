---
layout: default
title: "Word2Vec"
subtitle: 벡터 공간에서 의미적으로 유사한 단어들이 가깝게 위치하도록 학습하는 Word2Vec
permalink: /nlp/word2vec/
use_math : true
nav_order: 2
parent: nlp
---

# [Word2Vec] Efficient Estimation of Word Representations in Vector Space (2013)

## Intro

NLP 시스템들이 보통 단어를 atomic unit으로 보고 있다는 점에 대해 문제제기를 하여, word2vec 구조를 만들어낸 논문이다. 기존에 취했던 방식은 단어들 유사성에 대한 notion도 없이, 그냥 아예 각각 따로 심플하게 구성했다. 이는 단순하고 편하긴 하지만 데이터셋이 한정되어 있고 많은 데이터셋에서 아직 한계에 머무르고 있기에 새로운 구조가 필요하다. 

1. N-gram에 이어 가장 성공적이었던 구조는 신경망 구조였기에, 이를 차용한다. 
2. 두 가지 expectation을 바탕으로 한다. 
    1. similar word는 close to each other할 것이다.
    2. 단어들이 여러 유사성 정도를 가질 것이다 (multiple degrees of similarity) 

놀랍게도 word representation의 유사성은 단순한 구문 규칙를 훨씬 뛰어넘는다는 것이 밝혀졌다. 예를 들면, vector(king) - vector(man) + vector(woman) 을 하면 vector(queen)에 가장 가까운 결과가 나온다는 것이다. 

## Model Architectures

앞으로, 모델 훈련의 복잡도는 다음과 같이 표현된다. 

\[O = E \times T \times Q,\]

E는 training epochs, T는 훈련 세트의 개수이고 Q는 모델 구조에서 정의된다. 일반적으로 E = 3-50, T = 1 billion 정도를 취한다. SGD와 back propagation을 통해 학습된다. 

### Feedfoward Neural Net Language Model

모델은 input layer, projection layer, hidden layer, output layer으로 구성이 된다. 

**input layer**

해당 단어의 이전 N개의 단어들을 input으로 받는다. 각 단어들은 1-of-V coding을 통해 인코딩된다. 

**projection layer** 

input layer의 단어들을 D차원으로 projection한다. 즉, projection layer의 크기는 NxD이다. 여기서 N개 단어에 대해 동일한 projection layer을 사용한다. 

**hidden layer** 

밀집된 상태이기에 계산이 복잡할 수 있다. 

**output layer**

출력 단어 V에 대한 확률 분포를 계산한다. 

각 훈련 세트의 계산 복잡도는 다음과 같다. 

\[Q = N \times D + N \times D \times H + H \times V,\]

여기서 $ N\times D$는 input → projection, $N \times D \times H$는 projection → hidden, $H \times V$는 hidden → output 부분의 계산을 의미한다. 이때, 마지막 부분에서 output의 확률분포 계산을 위해 원래 단어 크기 만큼인 V 만큼 계산을 하게 되므로, 단어들의 수가 많으면 병목이 생길 수 있다. 

이를 줄이기 위해 Hierarchical Softmax이 제안된다. 이는 단어들을 Huffman binary tree로 표현한다. 빈도가 높은 단어를 더 짧은 코드로 표현함으로서 계산의 수를 확 줄여버린다. 

### Recurrent Neural Net Language Model

Recurrent NNLM은 feedforward NNLM의 한계점, 특히 맥락의 길이를 구체화할 필요성을 보완하기 위해 만들어졌다. 또한, 이론적으로 RNN들은 복잡한 패턴들을 더 효율적으로 표현할 수 있기 때문이다. RNN 모델은 projection 모델이 없고, input, hidden, output layer만 가지고 있다. RNN은 hidden layer을 time-delayed connection을 통해 계속 연결해나가서 short term memory를 가지게 된다. 

RNN 모델의 계산 복잡도는 다음과 같다. 

\[Q = H \times H + H \times V,\]

## New Log-Linear Models

계산 복잡도를 줄이기 위한 두 가지 구조를 제안한다.

지금까지 나온 모델들의 계산 복잡도는 hidden layer의 non-linear한 구조 때문이었다. 그렇다고 이 구조를 아예 배제하기에는 이 non-linear한 구조가 신경망 구조의 메인 포인트이기 때문에 그럴 수는 없다.

즉, 우리의 목표는 다음과 같다

1. neural network만큼 정확하지는 못해도,
2. 효율적으로 작동하는 구조를 찾자. 

### Continuous Bag-of-Words Model

처음으로 제안된 구조는 BoW이다. 이 구조는 non-linear hidden layer은 없고 projection layer이 모두 공유되는 형태이다. 즉, 모든 단어에 대해 same projection layer이 있는 것이다. 

그러나 Continuous BoW의 경우 현 단어 이후의 단어들도 이용을 하기 대문에, 태스크에서 좋은 성능을 보인다. 4개의 이전 단어와 4개의 이후 단어들을 사용하여 현재 단어를 예측하는 방식을 사용한다. 

### Continuous Skip-gram Model

두 번째 구조는 CBOW와 유사하지만, 반대 느낌이다. 맥락을 통해 현 단어를 예측하는 것이 아니라, 같은 문장의 다른 단어를 기반으로 단어의 분류를 최대화한다. 정확히 말하면, 현재 단어를 가지고 앞뒤 단어들을 예측한다. 그 예측하는 범위를 늘리면 당연히 질이 향상되겠지만 계산이 복잡해진다. 지금 목적은 계산 복잡도를 줄이기 위한 것이므로 적절한 지점이 필요하다.
