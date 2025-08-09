---
layout: default
title: "12 Basic of CNN"
parent: deep-learning
permalink: /dl/sa/standalone-12/
use_math : true
---

# [Standalone DL] 12 Lecture - # 20 Basic of Convolutional Neural Network

## Problem

MLP는 어떤 문제점을 가지고 있는가? 

가장 큰 문제는 parameter의 개수가 많다는 것이다. 모델 layer가 쌓이면서 파라미터의 개수가 증가하게 된다는 것이다 - `overfitting` !! 

- 모델 capacity가 늘어나게 되므로 시간도 오래걸리고, overfitting이라는 문제가 생기게 됨. 필요한 복잡도보다 더 복잡하게 예측을 하게 되기 때문

특히 어떤 weight들은 `meaningless`하다. 

- 예를 들어 이미지 인식에서, 3x32x32 이미지를 1x3072로 한 줄로 쭉 펴서 traning을 시키게 될 것
- 여기서 주변에 위치한 픽셀끼리는 연관이 있을 수 있지만, 맨 앞 픽셀과 맨 마지막 픽셀은 크게 관련이 없을 수 있음. 즉, 이들은 굳이 같이 볼 필요가 없을 것

## Hierarchical organization

### MLP Limitation

그렇다면 사람은 어떻게 이미지를 인식하고 있는가? 

MLP 역시 사람의 구조를 모방한 것이었음 (선, 원 등의 단순한 물체를 볼 때에 국한됨) 

- 각각의 뉴런들이 다른 뉴런들과 이어져 있음
- 그에 대한 weight와 activation을 취해서 동작하도록 함

그러나 이렇게 단순한 구조만으로 인간이 이미지를 인식하고 있을까? 좀 더 복잡한 인간만의 구조를 파악해서 이를 가져오자. 

- 이미지라는 것은 매우 복잡함. 사람은 이것을 한 번에 보는게 아니라, 간단한 정보들을 담당하는 각각의 뉴런들이 있다는 것을 알게 됨
- 뉴런은 hierarchical organization을 따르고 있음
    - 시각세포와 직접적으로 연관되어 있는 shallow한 영역에 있는 뉴런들 (빛, 선의 orientation 등)
    - LGN, V1 등 좀 더 deep한 영역에 있는 뉴런들 (선이 움직이는가, 선이 이어져있는가 끝나는가 등)
- 깊이 들어갈수록 더 복잡한 정보를 처리하고 있음. 깊어질수록 고차원적인 정보를 처리한다 !
    - simple cells, complex cells, hypercomplex cells 등

<img width="523" height="339" alt="Image" src="https://github.com/user-attachments/assets/e9ccb2a0-dd0c-4b8b-8341-f07876929733" />

### Fully Connected Layer (MLP) to Convolution Layer (CNN)

<MLP> 한 뉴런이 그 전 레이어의 모든 뉴런들과 연결되어 있음  

<img width="712" height="265" alt="Image" src="https://github.com/user-attachments/assets/2694374a-085f-4dd1-a7eb-dc97d64a7408" />

- 이미지를 일렬로 쭉 편 다음 (10 weight classification이라고 가정할 때) 10 x 3072 weights를 곱해서 1 x 10 output이 나오도록 함
- input의 row와 weight를 각각 dot product한 값이 결과라고 볼 수 있음

<CNN> spatial한 정보를 잃지 않기 위해 chunk를 도입 

<img width="664" height="318" alt="Image" src="https://github.com/user-attachments/assets/c2f8bdce-e928-47c9-8723-e7cc3f24b410" />
<img width="660" height="308" alt="Image" src="https://github.com/user-attachments/assets/9314e1df-ca7c-4650-bdbe-fd500824dce0" />
<img width="759" height="286" alt="Image" src="https://github.com/user-attachments/assets/0c8faded-56db-4275-a535-ebeb47272479" />

여러 개의 convolution filter을 가지고 진행한다 

- filter 개수를 몇 개를 하냐에 따라서 다음 레이어의 depth가 달라짐 ! ex) 6개의 5x5x3 filter → 다음 레이어의 depth는 6이 됨.
- 가로 세로의 경우 - filter의 크기 (5x5) 에 따라 변경될 것

### Convolution Neural Network

<img width="530" height="411" alt="Image" src="https://github.com/user-attachments/assets/ab2e8928-cbb3-4722-a558-2f37253100d0" />

- 각각의 filter의 역할 - 첫 레이어의 경우 visualization할 수 있지만, 두 번째 레이어부터는 visual하게 볼 수 있는 정보가 아님 (할 수는 있지만 눈으로 이해하기는 어려운 상태)
- low level을 보면, 각 뉴런마다 담당하는 모양들이 있는 것으로 보임.
- mid-level, high-level 등 점점 고차원으로 갈수록 눈으로 봐서는 판단하기가 어려운 모양들이 있음. 좀 더 깊이있고 복잡한 모양들을 탐지한다고 이해할 수 있음 !
- 깊은 레이어로 들어갈수록 더 세세한 모양을 탐지하게 될 수 있음. 
ex.

→ 뇌 구조를 좀 더 모방하고 있다고 볼 수 있음 

<img width="743" height="319" alt="Image" src="https://github.com/user-attachments/assets/4d5b8aaa-6639-400f-b094-1aaf47cfb7bf" />
<img width="783" height="338" alt="Image" src="https://github.com/user-attachments/assets/2ee51796-4953-466b-8fde-b54730169402" />

몇 칸씩 움직이느냐 를 stride라고 표현한다. 하나의 filter가 칸수를 옮겨가며 정보를 뽑아내는데, 이때 위의 그림에서 보면 7x7에서 3x3 filter을 stride=3으로 움직이다 보면, 한 번까진 가능하지만 두 번은 옮길 수 없다는 것을 알게 된다. 

이런 경우, zero-padding을 사용한다 ! 이미지의 외부에다가 0 행렬을 넣어서 9x9로 만들어준다. output layer가 정수로 나올 수 있도록 조정해주는 것 

<img width="800" height="379" alt="Image" src="https://github.com/user-attachments/assets/eb647269-b4b9-45b4-8194-177e05634244" />

- pooling layer
    - 고해상도의 이미지에서는 한 픽셀이 가지는 정보는 작을 것. downsampling을 통해 한 픽셀 값이 가지는 정보가 좀 더 함축적으로 표현되도록 이미지 사이즈를 줄여주는 과정을 거침. (일반적으로 절반으로 pooling ! 224x224 → 112x112)
    - `max-pooling` : 2x2 filter, strdie=2로 max pool하여 이미지를 절반으로 줄일 때, 해당 filter에서 max 값을 가지는 것을 그대로 가져오는 것. 값이 작은 것은 활성화되지 않았다는 뜻 - 그 정보는 어차피 필요도가 낮은 것이라는 논리 !
        
        ![image.png](attachment:5c4aae4c-3266-4b90-b8c2-9095862aa64b:image.png)
        
- FC (fully connected)
    - 마지막에 FC 즉 `MLP`를 씌움 : 데이터의 개수가 작고, 각각의 값이 중요하고 고차원적인 정보를 담고 있으며 이 정보로부터 classification을 해내야 할 때 MLP를 쓰는 것이 가장 좋음
    - convolution layer 및 pooling을 거치면서 점점 중요한 정보들만 담은 고차원적인 feature vector이 나온 것이기 때문에, 이 마지막에 대해서 모든 값을 다 보며 classification을 할 수 있도록 MLP 사용 !