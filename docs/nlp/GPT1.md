---
layout: default
title: "[GPT-1] Improving Language Understanding by Generative Pre-Training(2018)"
permalink: /nlp/gpt1/
subtitle: using unlabeled data 
use_math : true
parent: nlp
---

<br />

[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 

트랜스포머 논문보다 훨씬 친절하다. 

<br />

## 1. Intro

모델을 학습시킬 때 가장 많은 시간이 소요되는 것은 바로 data annotation, data curation인 것 같다. 실제로 아주 정제가 잘 되어 가공할 필요가 거의 없는 데이터란 존재하지 않고, 가공되어있다 한들 연구자들이 사용하려고 하는 라벨 / 분류와 정확히 일치할 수가 없기 때문이다. 가공되어 있어도 이정도인데, 보통 연구에서 사용하려고 하는 real data는 비가공 데이터로 전혀 정제되어 있지 않고, 형식조차 중구난방이다. 

이 논문에서도 이 부분을 supervised NLP에서의 한계점으로 찝으며 논의를 시작하고 있다. labeled data를 사용하기 위해서는 결국 annotation이 필요하며, 이를 하다가 끝난다는 것이다. 

<div class="callout">

아니 그러면 unlabeled data로 해보면 되는거 아니야? 

그게 어려운 이유는 다음과 같다.
- text representation을 학습할 때 가장 좋은 것이 뭔지 불확실하다. 
- 학습된 표현을 target task로 연결시킬 때, 합의된 가장 효과적인 방법이 없다. 

</div>

그래서 여기선 `semi-supervised approach` 를 제시한다. 즉, unsupervised pre-training과 supervised fine-tuning을 결합하는 것이다. 광범위한 task에 최소한의 adaption을 가지고도 커버할 수 있는 모델을 구현하는 것이 최종적인 목적이라고 할 수 있다. 이를 위해 두 단계의 학습을 수행한다. 

1. unlabeld data에 대해서 language modeling objective를 사용하여 initial parameter를 학습함 
2. 이를 target task에 적용하기 위해서 supervised learning 

모델의 구성 요소로는 transformer을 활용한다. transformer은 attention 구조를 활용하기에 long-term text와 diverse task에 transfer할 때 강력하기 때문이다. transfer을 할 때 `traversal-style approaches` 를 기반으로 하여, text input을 single contiguous sequence of token들로 구조화하는 작업을 진행한다. Pre-train된 모델에서 변형을 최소화하여 파인튜닝을 효과적으로 진행할 수 있다. 

<div class="callout">

Traversal-style approach가 뭔데? 

task가 자연어로 설명되어 입력될 때 그 입력을 토큰으로 구분하는 것. 보통 start token, end token, delimiter 등으로 이루어져 있다. 

</div>

<br />


## 2. Related Work

### 2-1. Semi-supervised Learning for NLP

초기에는 unlabeled data를 이용하여 word-level / phrase-level 을 계산한 후 이를 supervised model의 특성으로 이용하였다. 그리고 실제로 연구자들은 unlabeled copora에서 학습된 word embedding을 이용하여 성능을 개선해왔다. 그치만, 이들은 그저 word-level에서만 작동할 뿐, 우리가 주목하고자 하는 long-term과는 맞지 않는다. 그래서 최근 연구자들은 unlabeled 데이터에서 word-level 이상을 학습할 수 있도록 하는 인코딩을 목표로 연구하고 있다. 

### 2-2. Unsupervised Pre-training

unsupervised pre-trainin은 good initialization point를 잡는 것에 가장 큰 초점을 두고 있다 (supervised learning objective 수정하기 x). 각 task에 대해 모두 맞춤형으로 제작하는 것이 아니라, 하나의 모델이 최소한의 변경만 가지고 adapt할 수 있도록 하여 효과성을 극대화한다. 

<br />

## 3. Framework

### 3-1. Unsupervised Pre-training

Unlabeled 된 token $ U = \{u_1, ..., u_n\} $가 주어졌을 때, MLE를 위해 

$
L_1(U) = \sum_{i} \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta) \quad (1)
$

를 사용한다. 

k는 context window size이고, P는 conditional probability이다. 즉, **k개의 문맥 단위 창문으로 그 다음에 u_i가 나올 확률이 maximize되도록 \Theta 를 학습한다** , 라고 이해하면 될 듯 하다. i 번째 단어를 위해 (i-k)번째부터 (i-1)번째 단어까지를 보고, i 번째 단어가 나올 확률을 최대화하도록 한다는 것이다. MLE로 적용되어 loss function 기반으로 학습하게 될 것이다. SGD를 활용하기에 back-propagation도 진행한다.  

Language Model로는 multi-layer Transformer decoder을 이용하여 multi-headed self-attention을 진행한 후, position-wise feed-forward layer을 통해 target token에 대한 output distribution을 만든다. decoder만 이용한다는 점이 특징이다. 

$
h_0 = U W_e + W_p h_l = transformer\*block(h*{l-1}) \quad \forall i \in [1, n]
$

$
P(u) = softmax(h_n W^T_e) \quad  
$

$ U = (u_{-k}, ..., u_{-1}) $ 는 토큰의 context vector은 layer 수이다. 즉, context vector * token embedding + positional embedding으로 h0을 만들어서 transformer_block에 넣고, 그거랑 token embedding의 전치행렬을 곱한 것(점곱)을 softmax에 넣어서 확률값을 구한다. 이렇게 output distribution을 구한다는 것이다. 

### 3-2. Supervised fine-tuning

이렇게 initialization을 진행한 후, labeled dataset을 이용하여 parameter들을 supervised target task에 맞게 조정한다. input $ x_1, ..., x_m $과 그에 대한 label y가 있다고 생각할 때, input이 pre-trained model을 통과하며 activation $ h^m_l $를 얻고, W_y parameter과 함께 added linear output layer에 들어가서 최종적으로 y를 예측하게 된다. 

$
P(y | x_1, ..., x_m) = softmax(h^m_l W_y) \quad  
$

즉, x_1부터 x_m 까지의 입력값을 가지고 activation과 parameter을 곱해서 softmax를 취하여, x_1 ~ x_m 이 입력값으로 들어갔을 때 y 가 도출될 확률을 구하는 것이다. 

이후, 이를 모든 input-label에 대해 진행하여, 로그합을 구해 L_2(C)를 구한다. 

$
L_2(C) = \sum_{(x,y)} \log P(y | x_1, ..., x_m) \quad
$

직관적으로 생각하면, 모델이 y에 대해서 높은 확률을 예측했다면 L_2(C)가 큰 값을 가질 것이다. 학습 시 이렇게 최대화하는 방향으로 파라미터가 조정될 것이고, 이는 모델이 target task에 대해 높은 정확도로 예측을 한다는 의미이다. 즉, Loss Function이다. 

추가적으로 fine-tuning을 할 때, language modeling을 auxiliary objective로 추가하는 것이 좋다는 것을 발견했다. 그 이유는 다음과 같다. 

1. supervised model의 generalization 능력을 향상시킨다
2. convergence를 가속화한다 (빠르게 안정화된다는 뜻) 

그래서 unsupervised learning을 할 때 썼던 L_1(C)와 supervised fine-tuning할 때 만든 L_2(C)를 모두 사용한다 (L_1(C)에는 가중치를 붙인다). 

$
L_3(C) = L_2(C) + \lambda \cdot L_1(C) \quad (5)
$


<div class="callout">

L_1(C)에 가중치를 붙이는 이유?

모델 성능을 최적화하기 위함. L2는 이미 task에 최적화가 되어 있으므로, general하게 initialized 되어 있는 상태인 L1을 조정해서 효율성을 높인다. 두 L function의 균형을 조절하면서, 모델이 특정한 과제에 치우치지 않도록 하는 것이다. (그냥 가중치 가해서 더하기만 한다니 .. 직관적이면서도 이로 적합이 되는 것도 참 신기 ..) 

</div>

<br />

### 3.3 Task-Specific input Transformations

text classification과 같은 태스크에 대해서는 우리가 바로 fine-tuning하면 잘 수행해낼 수 있다. 근데 QA(질문에 답하기)나 TE(추론하기)는 구조화된 input이 있고(문장 짝, 문서, 질문-답), 우리의 pre-trained model은 아예 contiguous sequence 로 이루어져 있기 때문에, 이런 태스크까지 잘 수행해내기 위해서는 좀 수정이 필요할 것이다. 

**Textual entailment Task** 를 위해서, premise *p*와 hypothesis *h* 토큰 순서를 결합한다 (사이에 delimiter token 추가) 

**Similarity Task** 를 위해서, 두 문장의 쌍 순서가 정확하게 정해져 있는게 아니기 때문에 delimiter token을 추가한 뒤 각각 독립적으로 처리해서 두 개의 시퀀스 표현을 만든다. 

**Question Answering and Commensense Reasoning Task** 를 위해서, context document z, question q, 가능한 answer들 {a_k}를 가지고 결합한 뒤, softmax를 통해 output 분포를 계산한다.