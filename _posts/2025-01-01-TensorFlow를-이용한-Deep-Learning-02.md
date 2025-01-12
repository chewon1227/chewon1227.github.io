---
layout: post
title: TensorFlow를 이용한 Deep Learning 02
subtitle: Colab, TensorFlow를 이용해서 Deep Learning의 기초를 구현해보자 
categories: DL
tags: [colab, dl]
---


TensorFlow로 간단한 Linear Regression을 구현해보고자 한다. 

TensorFlow의 기본인 3 가지 단계를 거치면서 구현이 될 것이다

1. 그래프를 build한다
2. session을 run하면서 그래프를 실행시킨다
3. 실행 결과라 그래프를 업데이트하고 값을 반환해준다 

H(x) = Wx + b 꼴의 형태를 만들어보자. 

```python
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
```

여기서 이 데이터들은 trainable한 데이터들이다. 즉, 텐서플로우가 실행되면서 자체적으로 변경될 수 있는 값이라는 것이다. 

텐서플로우를 실행시키기 전에 먼저 shape (unit, dim) 을 지정해준다. 

```python
tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# units == output shape, input_dim == input shape
```

`tf.keras.Sequential()` 은 순차적 신경망 모델을 정의하는 데 사용된다. 레이어를 순서대로 추가하면서, 입력 데이터가 처음부터 끝까지 한 방향으로만 처리되는 구조이다 (CNN 등에 사용될 수 있을 것). 대신 branching, parallel 연결이 필요한 복잡한 모델은 정의할 수 없다는 점이 한계점이다. 

이제 gradient descent를 위해 `tf.keras.optimizers.SGD`를 생성해서, 신경망 학습 과정에서 가중치를 업데이트하는 방식을 정의한다. 이를 통해 학습 속도 등의 하이퍼파라미터를 설정할 수 있다. 

```python
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
tf.model.compile(loss='mse', optimizer=sgd)  
tf.model.summary()
```

여기서 SGD (Stochastic Gradient Descent ; 확률적 경사하강법)을 사용했다. 확률적인 이유는, 전체 데이터가 아니라 batch에 의해 선택된 소규모 데이터를 이용해서 가중치를 업데이트하기 때문이다. 계산 효율성도 높일 수 있고, 자꾸 지역 최적해로만 편중되지 않도록 만들어줄 수 있다. 

`tf.model.compile()` 을 통해 옵티마이저를 모델에 설정한다. 

`tf.model.summary()` 를 통해 결과를 확인해보면 다음과 같이 나왔다 

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| dense (Dense) | (None, 1) | 2 |

그러면 모델도 준비되었으니 이제 진짜 학습을 해보자 !!!

```python
tf.model.fit(x_train, y_train, epochs=200)
```

Epoch 1/200

**1/1** ━━━━━━━━━━━━━━━━━━━━ **0s** 443ms/step - loss: 8.2096
Epoch 2/200

**1/1** ━━━━━━━━━━━━━━━━━━━━ **0s** 106ms/step - loss: 3.7242
Epoch 3/200

**1/1** ━━━━━━━━━━━━━━━━━━━━ **0s** 58ms/step - loss: 1.7081

아주 잘 돌아가는 것을 볼 수 있다. 9초만에 끝난다 (이런 모델이 어디있담!!!!!) 

잘 학습이 되었다면 이 모델을 가지고 직접 예측값을 뽑아보자. 

```python
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)
```

**1/1** ━━━━━━━━━━━━━━━━━━━━ **0s** 39ms/step
[[-3.9989054]
 [-2.9994369]]

바로 이렇게 배열로 도출되는 것을 볼 수 있다. 즉, 5에 대한 모델의 예측값은 -3.9989054, 4에 대한 모델의 예측값은 -2.9994369인 것 ! 





[참고 유튜브 - hunkim](https://www.youtube.com/watch?v=mQGwjrStQgg&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=5)