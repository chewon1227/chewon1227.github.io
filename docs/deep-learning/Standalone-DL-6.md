---
layout: default
title: "06 Parameterize "
parent: deep-learning
permalink: /dl/sa/standalone-06/
subtitle: Dl and MLP 
use_math : true
---

# [Standalone DL] 06 Lecture - #11 Parameterize 

## Parameterize

### why?

- 다양한 변수를 한 곳에서 관리하기 위해서
- hyperparameter이 모델 안에 이미 fix되어 있지 않도록 하기 위해 (변수를 변경했다고 생각했는데 실제 결과가 차이가 없어서 중요하지 않은 변수라고 잘못 착각하게 될 수 있음)
- re-usability, readability
- 다양한 실험의 튜닝을 `auto hyperparameter optimizer`에게 제공하기 위해 !!

### Hyperparameter

- hyperparameter : non-trainable (자동 고정을 하고 출발하는 값들. 레이어 등..)
- parameter : trainable (loss function - backpropagation ..)

### Argparse

```python
import argparse 

parser = argparse.ArgumentParser()
args = parser.parse.args("")

args.num_layer = 5
print(arg.num_layer) # 5 
print(args) # Namespace(num_layer=5)

args.in_dim = 100
print(args) # Namespace(in_dim=100, num_layer=5)

linears = []
for i in range(args.num_layer):
		linears.append(i)
print(linears) #[0,1,2,3,4]

d = vars(args)
print(d, type(d)) # {'num_layer':5, 'in_dim':100} <class 'dict'>

```

즉, `argparse`는 딕셔너리 형태로 hyperparameter들을 지정할 수 있는 것

