---
layout: default
title: "TensorFlow를 이용한 Deep Learning 04"
permalink: /dl/tf/tensorflow-04/
subtitle: Colab, TensorFlow를 이용해서 multi-variable linear regression을 해보자. 
parent: deep-learning
---


이제 multi-variable linear regression을 해보자. 말 그대로, 변수가 x 하나만 있는게 아니라 여러 개 있는 형태이다. 조금 더 현실과 가까워졌다고 생각하면 좋을 것 같다. 

필요한 패키지들을 설치해주고 ,, 임포트해온다 

```python
import tensorflow as tf
import numpy as np
```

배열 형태로 x 여러 개에 대해 y 1개가 나오는 multi-variable 형태를 가져온다. 

```python
x_data = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
y_data = np.array([[152.],
          [185.],
          [180.],
          [196.],
          [142.]])

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))
tf.model.add(tf.keras.layers.Activation('linear'))

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))
tf.model.summary()
```

`tf.keras.Sequential()` 로 모델을 만들어준다. 

```python
history = tf.model.fit(x_data, y_data, epochs=100)
```

`tf.model.fit`을 통해 모델을 훈련한다.

•••

Epoch 95/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

45ms/step - loss: 9.0829
Epoch 96/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

62ms/step - loss: 9.0787
Epoch 97/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

56ms/step - loss: 9.0744
Epoch 98/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

47ms/step - loss: 9.0702
Epoch 99/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

56ms/step - loss: 9.0659
Epoch 100/100

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

50ms/step - loss: 9.0617

 

```python
y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
```

**1/1**

━━━━━━━━━━━━━━━━━━━━

**0s**

37ms/step
[[161.5736]]

결과가 잘 도출되었다.





[참고 유튜브 - hunkim](https://www.youtube.com/watch?v=fZUV3xjoZSM)