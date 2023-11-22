# TensorFlow는 구글에서 만든 딥러닝 프로그램을 쉽게 구현할 수 있도록 기능을 제공하는 라이브러링이다.
# 텐서플로 자체는 기본적으로 C++로 구현이 되나 , 파이썬 ,자바 , 고(go) 등 다양한 언어를 지원한다.
# 하지만 파이썬을 최우선으로 지원하며 대부분의 편한기능들을 파이썬 라이브러리로만 구현 되어 있어서 python으로 개발하는것을 추천한다. 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.__version__)
print('GPU사용가능!~!' if tf.test.is_gpu_available() else 'GPU사용 불가능 ㅠㅠ')