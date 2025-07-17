---
layout: post
title: "[ViT] An Image is Worth 16x16 Words : Transformers for Image Recognition at Scale (2021)"
permalink: /nlp/ViT/
subtitle: Transformer in Image Recognition 
use_math : true
parent: nlp
---



https://arxiv.org/abs/2010.11929

vision 분야에서의 transformer 모델이다. 


## 1. Introduction

Self-attention-based 구조들, 특히 transformer와 같은 것들은 NLP에서 모델의 선택지 중 하나가 되었다. 코퍼스에서 pre-train된 것을 가지고 작은 특정한 데이터셋으로 파인튜닝하는 것이 가장 널리 알려진 접근 방법이다. transformer을 이용하면 100B가 넘는 파라미터 사이즈에도 잘 학습을 해낼 수 있다. 

근데 computer vision에서는, convolutional 구조가 아직 지배적이다. 그래서 CNN을 self-attention과 결합해보려는 시도가 나오고 있었긴 하다. 이미지 인식에서 sota 모델은 ResNet이었다. 

transformer이 NLP에서 성공을 거둔 것을 보고, 연구자들은 tarnsformer을 그대로 - 최대한 조금만 변경해서 - 이미지에 적용해보기로 했다. 그러기 위해 transfomer의 인풋으로 이미지를 패치 단위로 쪼개고 이 패치들 (image patch)에게 선형적 순서를 가진 embedding을 부여했다. Image patch 들은 NLP에서의 토큰처럼 다루어진다. 

ImageNet과 같은 중간 사이즈의 데이터셋으로 강한 정규화 과정 없이 학습을 시킬 때, 이 모델들의 성능은 ResNet보다 아주 조금 떨어지는 수준의 성능을 가진다. 좀 안 좋은 결과를 예상하게 될 수도 있는데 : Transformer들은 CNN에 비해 `inductive bias`가 좀 부족할 수밖에 없다. 

<div class="callout">

`inductive bias`

- 주어지지 않은 입력의 출력을 예측하는 것.
- 보지 못한 데이터에 대해서도 귀납적 추론이 가능할 수 있도록 알고리즘이 가지고 있는 가정들의 집합 !
- 일반적으로 모델이 가지는 generalization problem : brittle하다는 점과 spurious하다는 점. 이것을 해결하기 위한 하나의 방법임

</div>

그러나, 모델을 더 큰 데이터셋으로 학습을 시키게 되면 달라진다. 연구자들은 large scale training이 inductive bias를 넘는다는 것을 발견했다. ViT는 충분한 크기로 사전훈련이 된 후 데이터 포인트가 작은 작업으로 전환될 때 우수한 결과를 얻는다. ImageNet-21k 데이터셋이나 JFT-300M 데이터셋으로 사전훈련되었을 때, ViT는 sota 모델의 성능 정도이거나 그를 능가한다. ImageNet에서는 88.55%, ImageNet-Real에서는 90.72% 등등의 성능을 가진다. 

<br />
<br />

## 3. Method


### 3-1. Vision Transformer

모델 구조는 다음과 같다. 

<img width="555" alt="Image" src="https://github.com/user-attachments/assets/5532e92b-d8d4-471b-b38c-8ea3ab010b8d" />

<br />

Transformer Encoder을 먼저 보자. 

transformer은 1D token embedding을 인풋으로 받는다. 2D 이미지를 다루기 위해서 이미지를 flatten 2D patch로 reshape한다. 

$ x ∈ R^(H×W ×C) → x_{p} ∈ R^ (N ×(P^2 x C)) $ 

이때 (H,W)는 원래 이미지의 해상도, C는 채널의 수, (P, P)는 각 이미지 패치의 해상도이다. 즉, N = HW/P^2 가 패치들의 개수가 될 것이다 (이것은 transformer의 효과적인 시퀀스 길이에도 영향을 미친다). transformer은 모든 레이어에 대해서 정해진 벡터 사이즈 D를 사용하므로, 연구자들도 모든 패치들과 맵을 D차원으로 눌렀다. 이 projection을 `patch embedding`이라고 한다. 

BERT의 `class token`과 비슷하게, embedded patch에다가 학습 가능한 embedding을 추가한다. 이때, 이 patch는 transforemr encoder의 출력 상태의 이미지 representation으로 사용된다. 
classification head는 하나의 hidden layer을 가진 MLP에 의해 구현되고, fine-tuning time에는 single linear layer로 구현된다. 

<div class="callout">

BERT의 `class token` 
<br />
BERT의 [CLS] token은 문장 전체의 의미를 대표하는 특수 토큰. 문장 분류 작업에서 중요한 역할을 한다. 
<br />
<br />
input : I love this movie  <br />
BERT sequence : [CLS] I love this movie. [SEP]  <br /> 
output[0] : [CLS]에 해당하는 임베딩 (문장 전체 요약) <br />
output[1] ~ : 각 단어의 임베딩

</div>

position embedding 또한 patch embedding에 들어갔다. 


**Inductive Bias** 

ViT가 CNN보다 image-specific한 inductive bias가 적다는 것은 사실이다. <br />
CNN에서는 two-dimensional neighborhood structure와 translation 등분산이 전체 모델의 각 레이어에 모두 들어가게 된다. 그러나 ViT에서는 MLP 레이어만 지역적이면서 tranlationlly하게 등분산이고 self-attention은 global하다. <br />
즉, 정리하자면 CNN은 spatial information을 구조 내에 내장해둠. 주변과 비교한 local 정보가 전체 모델에 싹 들어가있음. <br />
ViT에서 MLP는 CNN처럼 로컬하게 작동하지만, self-attention은 전체 이미지를 한 번에 보므로 패치 간 global한 관계를 학습함. <br />
Two-dimensional neighborhood structure (ex. 픽셀 주변 정보. local 정보)은 매우 드물게 사용되는데, <br />
- 모델 초기에 이미지를 패치로 만들고 다양한 해상도를 가진 이미지를 positioning하기 위해 사용됨 
- position embedidng을 조정하기 위한 fine-tuning 때 사용됨 
이 외에는 초기화 시점의 position embedding은 2D 패치들의 위치에 대한 정보가 없고, 패치 간 모든 spatial relation들을 처음부터 학습해야 한다. 

<div class="callout">
CNN은 공간적 구조가 내장되어 있으나, ViT는 공간 정보가 없고 대부분 학습을 통해 직접 알아내야 한다 ! <br />
ViT는 위치 개념이 없기 때문에 로컬 구조를 기본적으로 인식하지 못해서, 같은 정보를 학습을 통해 알아내야함. 
</div>

<br />

**Hybrid Architecture** 
raw image patch의 대체제로, input sequence는 CNN의 feature map에서 나올 수 있기 때문에 꼭 이미지를 자른 패치를 이용할 필요는 없다. <br />
이 hybrid model에서, CNN feature map을 다시 patch로 나누고, 각 patch에 patch embedding을 투영해서 trnasformer이 이해할 수 있는 시퀀스로 만든다. transformer은 고정된 길이의 벡터 시퀀스를 입력으로 받기 때문 ! <br />
특별한 경우로 patch는 1x1 사이즈를 가질수도 있다 - input seqence는 feature map을 그냥 flatten한 뒤 transformer 차원에 projection한 단순한 과정으로 얻을 수 있다는 것. 그냥 펼치면 된다. 

<br />

### 3-2. Fine-tuning & Higher Resolution 

<br />

일반적으로는 ViT를 큰 데이터셋에다가 사전학습을 시키고, 작은 downstream task에 맞추어서 파인튜닝을할 것이다. <br />
근데 연구자들은 pre-trainined prediction head를 제거하고 (!!!!) zero-initialized D x K feedforward layer을 추가한다. 대체왜 ??? <br />
**higher resolution에서 파인튜닝하는 것이 사전학습하는 것보다 효과적** 이기 때문이다 .. <br />
해상도가 커져도 patch size는 그대로 유지되므로 - 패치 개수가 늘어남 - 시퀀스 길이가 더 길어지기 때문 ! ViT는 시퀀스 길이가 달라도 동작은 잘 된다. <br />
그런데 pre-trainined된 position embedding은 시퀀스 길이가 늘어나기 전이므로, 이제 유의미하지 않을 것. 그래서 기존의 position embedding을 2D 보간해서 늘려준다. <br />
이것이 바로 ViT가 inductive bias를 넣는 부분. positioning embedding을 2D grid structure로 조정하는 파트이다 ! 


<br />
<br />

## 4. Experiments 

<br />
ResNet, ViT, hybrid의 학습 능력을 평가한다. 각 모델의 요구사항에 맞춰서 다양한 데이터셋과 다양한 벤치마크들을 이용한다. 

### 4-1. Setup
**Dataset** 
데이터셋으로는 ILSVRC-2012 ImageNet 데이터셋과 ImageNet-21k, JFT를 이용한다. 

**Model Variants**
ViT configuration은 다음과 같다. 
<img width="451" alt="Image" src="https://github.com/user-attachments/assets/6549ef4f-2ebb-4b7a-92bd-97b26454cd31" />
patch size가 더 작을수록 더 많은 patch가 생기고 시퀀스 길이가 길어진다. 즉, 작을수록 성능은 좋을 수 있어도 계산량이 증가한다는 것. 

비교용으로 ResNet을 쓰는데, 그대로 안쓰고 약간 수정한다 - ResNet(BiT) . <br />
이런거 막 수정해도 되는거야? 싶긴한데, 다음 두 가지 사항이 적용된 것이다.  <br />
- batch normalization 부분을 group normalization으로 바꾼다 
- standardized convolution을 사용한다 
-> 결론적으로 transfer 성능이 올라갔다. 즉, 공정한 비교를 위해 ResNet도 transfer에 최적화를 시켜준 것. 

<br />
### 4-2. Comparison to SOTA
<img width="649" alt="Image" src="https://github.com/user-attachments/assets/c59d01bb-ebd8-445a-ac89-16c34d2e4985" />
작은 ViT-L/16 모델은 BiT-L 모델보다 잘한다. 큰 모델인 ViT-H/14는 훨씬 잘한다. 

<br />
### 4-3. Pre-training data Requirements
ViT는 JFT-300M 데이터셋과 같이 큰 데이터셋으로 사전학습했을 때 성능이 좋다. 이에 기반해서, 데이터셋 사이즈가 얼마나 결정적으로 작용하는지 알아보기 위한 실험을 진행했다. <br />
<img width="647" alt="Image" src="https://github.com/user-attachments/assets/2842013a-5714-4021-a19f-5be77e15d44e" />
결론적으로는 데이터셋이 많아질수록 확실히 성능이 올라간다. ResNet(BiT)의 경우 비교적 일관적이지만 ViT는 영향을 꽤 받는 것 같다. 

### 4-5. Inspecting ViT
ViT의 구조를 제대로 이해하기 위해, 몇 가지 실험을 진행했다. 
<img width="658" alt="Image" src="https://github.com/user-attachments/assets/4f3bd4fd-7b81-40d7-a885-c7bf18854445" />
각 filter의 기능이, low dimension의 CNN filter 기능과 유사했다. 또한, ViT는 projection 후 positional embedding을 patch representation에 추가한다. 가까운 패치 간의 유사도가 높아지므로 input patch 간의 spatial information이 잘 학습될수밖에. 
self-attention을 통해서 이미지 전체의 정보 통합이 가능한지 확인한 결과 - attention distance는 receptive field처럼, 낮은 레이어의 self-attention head는 CNN과 같이 지역적 localizaation 효과를 보였다. 


<br />
<br />

## 5. Conclusion 
large dataset에서 잘 작동하는, 이미지 분류에서의 transformer 모델 ViT. 

<br />
<br />

<div class="callout">
그렇다면 ViT를 쓰려면 대규모 사전학습 데이터로 한번 돌린 다음에 사용해야 하는데. 이게 쉽지 않을 것으로 보임. 사전학습 데이터에 왔다갔다 하지 않는 더 좋은 구조가 나오길. 
그러려면 다음과 같은 조건이 필요하지 않을까? <br /> <br />
1. Strong inductive bias <br />
2. flat 하지 않게 계층적으로 정보를 다루는 것 - CNN처럼 feature map을 점점 줄여나가는 형식. 저차원 -> 고차원. 
3. position embedding이 없어도 spatial information을 잘 습득할 수 있게 하면 되지 않을까. -> Swin 구조? 이 논문도 읽어봐야겠다. 