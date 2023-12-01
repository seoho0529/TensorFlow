import tensorflow as tf
import sys
import numpy as np
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])  # 0번째 feature
print(y_train[0])  # 0번째 label

# cnn은 채널(channel)을 사용하므로 3차원 4차원으로 변환
x_train = x_train.reshape((60000, 28, 28, 1))  # (-1, 28, 28, 1) 이라 쓰면 알아서 판단해줌
x_test = x_test.reshape((10000, 28, 28, 1))  # (-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
print(x_train[:1])

x_train = x_train / 255.0
x_test = x_test / 255.0

# 모델 (CNN : 고해상도, 크기가 큰 이미지를 전처리 후 작은 이미지로 변환 후 ==> Dense(완전연결층으로 전달)로 분류 진행)
input_shape = (28, 28, 1)

print('방법3 : Subclassing API사용')
import keras
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()  # 생성자 호출
        self.conv1 = keras.layers.Conv2D(filters=16, kernel_size=[3,3], padding='valid', activation='relu') # kernel_size 구조는 리스트, 튜플로도 가능
        self.pool1 = keras.layers.MaxPool2D((2,2))
        self.drop1 = keras.layers.Dropout(0.3)
        
        self.conv2 = keras.layers.Conv2D(filters=16, kernel_size=[3,3], padding='valid', activation='relu') # kernel_size 구조는 리스트, 튜플로도 가능
        self.pool2 = keras.layers.MaxPool2D(2,2) # (2,2)는 높이2, 너비 2차원의 풀 크기
        self.drop2 = keras.layers.Dropout(0.3)
        
        self.flatten = keras.layers.Flatten(dtype='float32')
        
        self.d1 = keras.layers.Dense(64, activation='relu')
        self.drop3 = keras.layers.Dropout(0.3)
        
        self.d2 = keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.drop1(net)
        
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.drop2(net)
        
        net = self.flatten(net)
        
        net = self.d1(net)
        net = self.drop3(net)
        
        net = self.d2(net)
        return net
        
model = MyModel()
temp_inputs = keras.layers.Input(shape=(28,28,1))
model(temp_inputs)
print(model.summary())

        