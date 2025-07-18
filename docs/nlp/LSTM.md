---
layout: default
title: "[LSTM] Long Short Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling (2014)"
subtitle: cell state, gate 구조를 통해 장기 의존성을 학습하는 LSTM
permalink: /nlp/lstm/
use_math : true
parent: nlp
---



## Intro

Speech는 복잡한 시계열 데이터이다. 참조하는 맥락도 직전의 좁은 맥락일 때도 있고, 이전의 넓은 맥락일 때도 있을 만큼 제각각이다. 바로 직전 명사를 대하는 대명사일 수도 있지만, 대화하는 큰 맥락을 뜻할 수도 있다. 

Speech Recognition을 위한 `Acoustic Modeling`에서는 DNN이 SOTA였는데, 사실 한계가 있다. fixed-size sliding window를 받기 때문에 정해진 양씩만 볼 수 있는 것. 말하는 속도가 다를 시 인식하기 어렵고 long-term을 커버하기 어렵다. 

```
Acoustic Modeling에서는 뭐가 중요할까? 

- 음성 신호에서 입력 특징을 추출하는 것 
- 음성의 비선형적 특징
- 시계열 데이터 
```

RNN은 previous step을 인풋에 포함해서 current step을 예측하기 때문에, 또 그 맥락을 내부 state에 저장하고 있기 때문에 DNN과는 다르다. 

|  | DNN | RNN |
| --- | --- | --- |
| 기본적 구조 | Fully Connected  | 순환구조 신경망 |
| 데이터 처리 방식 | 입력 데이터를 독립적으로 처리함. 시간 연속성 X  | 이전 단계의 출력을 다음 단계의 입력으로 사용함 |
| 메모리 | 현 레이어 출력만 다음 레이어로 전달함  | 이전 단계의 출력 (hidden state)를 전달. 장기 의존성 학습 가능  |
| 한계점 | sequential한 연관성을 처리하기 어려움 | sequence가 길어지면 기울기 소실 |

LSTM은 `acoustic modeling` 태스크 성능을 높이는 범위 내에서 RNN의 한계를 극복한다. 

1. Context-free, Context-sensitive language 를 잘 학습한다 
    - context-free language : 문맥 자유 문법 (CFG) 이 생성할 수 있는 언어의 집합. G = (V, \Sigma, R, S) 로 규칙이 정의된다. 문법적으로 구조화된 언어를 모델링할 수 있지만, 복잡한 문백 의존적 언어는 처리하기 어렵다.
    - context-sensitive language : 문맥 민감 문법 (CSG) 이 생성할 수 있는 언어의 집합. \alpha A \beta \to \alpha \gamma \beta fh 로 규칙이 정의된다. 문맥 의존적인 구조를 다룰 수 있지만, 복잡도와 연산 비용이 상대적으로 높다.
2. Bidirectional LSTM은 입력 시퀀스를 양방향으로 처리하여 성능 향상을 입증한다 
    - 주로 CTC Layer과 함께 사용되고, 나누어지지 않은 sequence 데이터를 학습한다.
3. Deep Bidirectional LSTM은 hybrid speech recognition에서 DNN보다 더 좋은 성능을 보인다. 

## LSTM Network Architectures

### Conventional LSTM

recurrent hidden layer에 `memory blocks` 를 포함한다. memory block은 memory cell들을 가지고 있고, 그 상태의 데이터들을 저장하고 있는 self-connection들을 포함한다는 점이 특징이다. 이는 gate처럼 flow of information을 조절하는 역할을 한다. 각 memory block은 input gate와 output gate, forget gate를 포함하고 있다. 

- input gate : memory cell로 들어가는 flow of input activation을 조절한다.
- output gate : output flow of cell activation을 조절한다.
- forget gate : cell의 input으로 들어가기 전에 cell의 internal한 state를 스케일하여, 그 cell memory를 forget할지 reset할지 결정한다. 쉽게 말하면, memory cell 데이터를 잊을지 말지 선택하는 것이다.
- peephole connection : 같은 cell 내에서, internal cell에서 gate로 가는 연결. output의 정확한 타이밍을 학습한다. 이전 cell state를 gate에 적용해서 더 많은 맥락을 고려하게 해준다. 항상 성능을 향상시키는 것은 아니기에, 적용해보면서 확인해야 된다.

LSTM의 핵심은 network가 장기 상태에서 읽어들일 부분, 기억할 부분, 삭제할 부분을 명확히 알고 학습하는 것이다. long-term memory c_{t-1} 은 forget gate를 지나면서 기억을 일부 잊고, input gate로부터 새로운 정보를 추가하고, output gate으로 전달되어 단기기억 상태인 h_t과 셀의 출력 y_t를 만든다. 

LSTM은 시간 데이터에 따른 equation을 기반으로, unit activation을 계산해서 input sequence x → output sequence y 로의 매핑을 연산한다. 

\[
i_t = \sigma(W_{ix} x_t + W_{im} m_{t-1} + W_{ic} c_{t-1} + b_i) \tag{1}, 
f_t = \sigma(W_{fx} x_t + W_{fm} m_{t-1} + W_{fc} c_{t-1} + b_f) \tag{2}, 
c_t = f_t c_{t-1} + i_t g(W_{cx} x_t + W_{cm} m_{t-1} + b_c) \tag{3}, 
o_t = \sigma(W_{ox} x_t + W_{om} m_{t-1} + W_{oc} c_t + b_o) \tag{4}
m_t = o_t h(c_t) \tag{5}, 
y_t = \phi(W_{ym} m_t + b_y) \tag{6}
\]

1. W는 가중치를 나타낸다. 
    - W_{ix} 는 input gate에서 input으로 갈 때의 가중치 행렬
    - W_{ic}, W_{fc}, W_{oc} 는 peephole connection의 가중치 대각행렬
2. i, f, o, c는 각각 input, forget, output gate, cell activation vector 을 가리킨다 

### Deep LSTM

Deep LSTM은 단순한 LSTM 네트워크를 여러 계층으로 쌓아올려 더 복잡한 패턴과 시계열 관계를 학습할 수 있도록 설계된 구조다. 일반적으로 깊은 신경망은 더 강력한 표현력을 가지며, 이는 복잡한 음성 데이터와 같은 고차원 시계열 데이터에서 더욱 유리하다.
