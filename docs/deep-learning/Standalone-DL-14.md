---
layout: default
title: "14 Basic of RNN"
parent: deep-learning
permalink: /dl/sa/standalone-14/
use_math : true
---

# [Standalone DL] 14 Lecture - # 26 Basic of Recurrent Neural Network 

## Sequential Data

sequential한 데이터를 다루는 경우는 다음 네 가지가 있다. 

<img width="704" height="237" alt="Image" src="https://github.com/user-attachments/assets/27ce5fd9-ed87-4de8-ba7e-c05121c34b6c" />


1. `one-to-one` 
2. `one-to-many` : 이미지를 주고 캡션을 생성하는 것. CNN을 통해 이미지의 feature vector을 구한 후 인풋으로 넣어서, 이를 바탕으로 feature들을 설명하는 자연어 문장들을 생성하는 방식 
3. `many-to-one` : 다양한 sequence를 보고 어떠한 것을 예측하는 것. feature vector을 보고 classification을 수행하는 방식
4. `many-to-many` 
    - 여러 인풋을 받고 나서 하나의 스페이스를 만든 후 여러 아웃풋을 내놓는 방식 - 대표적으로 번역. 단어들이 들어갔을 때 문장에 대한 feature vector이 생성될 것이고, 이를 다른 언어의 단어들로 바꾸는 방식.
    - 여러 인풋을 받을 때 바로바로 아웃풋들을 내놓는 방식.

풀려고 하는 task가 이 네 가지 중 어떤 것인지 생각하고 행동하는 것이 좋다. 

<br />    

## Recurrent Neural Network

### RNN concept

단어들의 전후 맥락을 고려하기 위해 output of previous input과 new input을 같이 처리하자 ! 

<img width="654" height="229" alt="Image" src="https://github.com/user-attachments/assets/d3499d35-54ab-478a-bc8b-f26215d131d8" />


- 처리한 정보들을 다 가지고 있는 $ h_{t+3} $ 은 나름 sequence를 담고 있다고 생각할 수 있을 것 !

그렇다면 previous input과 new input을 그냥 합치는 것인가? 

<img width="752" height="251" alt="Image" src="https://github.com/user-attachments/assets/6044b6c7-2424-4131-9164-13dd48c1e317" />


- $ h_{t-1} $ 에는 가중치 $ W_h $ 를 곱해주고, $ x_t $ 에는 가중치 $ W_x $ 를 곱해줌
- 두 결과를 합친 후, 이 값을 non-linear activation (ex. tanh, ReLU) 에 통과시켜서 새로운 $ h_t $ 를 만듦
- 이 값이 다음 시점의 input으로 들어가서 같은 과정을 반복함

input / output 길이를 다양하게 하기 위한 방법 

- input - sequence를 다양하게 처리하는 방법
    1. max_token을 정해두고, 그것을 넘는 토큰들은 그냥 0으로 만들어버림 (그것을 위한 특별한 토큰을 만듦 - <blank> ) 
    2. 길이가 다양한 sequence들을 batch로 한 번에 처리하며, 짧은 것들이 끝날 때마다 마지막 리스트에 저장하고 더 이상 계산하지 않는 방식을 취함 (dynamic하게 진행) 
- output decoding의 길이를 다양하게 처리하는 방법
    1. token 개수를 정해둠 
    2. `<bos>` (begin of sentence) , `<eos>` (end of sentence) 라는 토큰을 달아줌 - <eos> 토큰을 감지한 뒤 끝나게 하면 됨 

### RNN Application

1. `one-to-one` 
    - 
2. `one-to-many` 
    - $ x_t $ 에다가 CNN을 통과한 이미지 벡터를 넣어줌 (CNN을 통한 encoding)
3. `many-to-one` 
    - 끝에 모델 한 개를 더 붙임. MLP나 가장 단순한 classification 모델을 붙이면 알아서 잘 분류를 해줄 것 !
4. `many-to-many` 
    - 번역과 같은 경우
        
        <img width="779" height="260" alt="Image" src="https://github.com/user-attachments/assets/77e64838-9ded-4e8b-aa8b-9c77b3d713fc" />
        
        - encoder - decoder 구조가 됨 (서로 다른 파라미터를 사용). feature을 학습하고 이해하되, 내뱉는 구조는 따로 있다
    - 여러 단어 분류와 같은 경우
        - 각각의 $ h_t $에 대해 classification 모델을 붙인다
        - 각 시점에 대한 y의 true 값과 모델에서 배출한 값이 나올 것이고 - 이걸 가지고 binary/multi classification의 방식대로 loss를 구한 후 back-propagation을 통해 계산하면서 optimizing하는 과정을 거치면 됨

### RNN with Math

<img width="626" height="353" alt="Image" src="https://github.com/user-attachments/assets/74c41388-81e5-4674-80b5-bbb51e97d806" />