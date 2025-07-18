---
layout: default
title: "TensorFlow를 이용한 Deep Learning 01"
permalink: /dl/tensorflow-01/
subtitle: Colab, TensorFlow를 이용해서 Deep Learning의 기초를 구현해보자 
parent: deep-learning
---

텐서플로우란?

- 구글에서 만든 오픈 소스 라이브러리이다. data flow graph를 사용해서 numerical computation이 가능하고, python 언어를 가지고 구현할 수 있다.
- data flow graph는, node가 operation이고 edge는 데이터(array/tensor)인 것!

```bash
pip install --upgrade tensorflow
```

```python
import tensorflow as tf 
tf.__version__ #2.18.0 
```

버전이 나오면 잘 설치가 된 것이다.


[참고 유튜브 - hunkim](https://www.youtube.com/watch?v=-57Ne86Ia8w&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=3)