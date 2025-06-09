---
layout: post
title: TensorFlow를 이용한 Deep Learning 03
subtitle: Colab, TensorFlow를 이용해서 Deep Learning의 기초를 구현해보자 
categories: DL
---


TensorFlow로 구현한 Linear Regression의 cost를 최소화하는 알고리즘을 구현해보자. 

필요한 패키지들을 설치해주고 ,, 임포트해온다 

```python
!pip install -U pip
!pip install -U matplotlib

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```

```python
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
tf.model.compile(loss='mse', optimizer=sgd)
tf.model.summary()
```

지난번처럼 데이터를 가장 기본적으로 설정을 해준 뒤, `tf.keras.Sequential()` 로 모델을 만들어준다. 

```python
history = tf.model.fit(x_train, y_train, epochs=100)
```

`tf.model.fit`을 통해 모델을 훈련한다. x_train, y_train이 입력/출력 데이터이고, epochs=100 즉 전체 데이터를 100번 반복학습한다. `history`에는 학습 과정에서의 loss가 저장된다. 

```python
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)
```

학습된 모델을 통해 새로운 데이터 (입력값 [5,4]) 에 대한 예측값을 계산하도록 한다. 

그리고 이제 loss의 패턴을 보려고 한다 

```python
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

epoch가 커질수록 loss가 줄어드는 형태를 확인할 수 있다.


[참고 유튜브 - hunkim](https://www.youtube.com/watch?v=Y0EF9VqRuEA&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=8)