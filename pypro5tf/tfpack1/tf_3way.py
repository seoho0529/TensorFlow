# 3 ways to create a Keras model with TensorFlow 2.0(Sequential, Functional, and Model Subclassing)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np

# 공부시간에 따른 성적결과 예측
x_data = [[1.],[2.],[3.],[4.],[5.]] # 이렇게 직접 2차원으로 만드는 것이 일반적

x_data = np.array([1,2,3,4,5], dtype=np.float32)
y_data = np.array([11, 39, 55, 65, 70], dtype=np.float32)
print(np.corrcoef(x_data, y_data))


print('1) Sequential api 사용 : 가장 단순한 방법, 레이어를 순서대로 쌓아올린 완전 연결층 모델 생성')
model=Sequential()
model.add(Dense(units=2, input_dim=1, activation='relu')) # 출력층 # input_dim=1은 input_shape=(1,)로도 가능
model.add(Dense(units=1, activation = 'linear'))
print(model.summary())


opti = optimizers.Adam(learning_rate=0.1)
model.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)


print('loss metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 ' , model.predict(x_data).ravel())


new_data=[1.5,2.3,6.8,8.0]
print('새 예측값 : ', model.predict(new_data).flatten())




import matplotlib.pyplot as plt
plt.plot(x_data.flatten(), model.predict(x_data), 'b', x_data.flatten(), y_data, 'ko')
plt.show()


# mse의 변화량
plt.plot(history.history['mse'], label='mean squared error')
plt.xlabel('epoch')
plt.show()


print('2) functional api 사용 : 유연한 구조로 설계 가능.')
# 다중입력값 모델을 만들 수 있음 concat

from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(1,))
# outputs = Dense(units=1, activation='linear')(inputs)  # 이전 층을 다음 층 함수의 입력으로 사용하도록 함
output1 = Dense(units=2, activation='relu')(inputs)
output2 = Dense(units=1, activation='linear')(output1)

model2 = Model(inputs, output2) # 입력,최종출력


opti = optimizers.Adam(learning_rate=0.1)
model2.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model2.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss metrics : ', loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 ' , model2.predict(x_data).ravel())


new_data=[1.5,2.3,6.8,8.0]
print('새 예측값 : ', model2.predict(new_data).flatten())



print('3) sub classing 사용 : 동적인 구조로 설계 가능. 나이도 높은 네트워크 처리 가능')
x_data = np.array([[1.],[2.],[3.],[4.],[5.]], dtype=np.float32)
y_data = np.array([11, 39, 55, 65, 70], dtype=np.float32)


print('**'*20)
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation='linear') # linear,relu 상관없음, 한개가 들어와서 두개가 빠져나감
        self.d2 = Dense(1, activation='linear') # 한개로 빠져나감


    def call(self, x):       # process 담당 | compile, fit... 등등 일때 자동으로 호출됨.
        inputs = self.d1(x)
        return self.d2(inputs)

model3 = MyModel()

opti = optimizers.Adam(learning_rate=0.1)
model3.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model3.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss metrics : ', loss_metrics)

print('설명력 : ', r2_score(y_data, model3.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 ' , model3.predict(x_data).ravel())
print(model3.summary())


print('3) sub classing 사용2')
from keras.layers import Layer
class Linear(Layer):  # Layer를 상속받음
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        # 모델의 가중치 관련 작업 기술
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                            initializer='random_normal',  trainable=True) # trainable=True 역전파 수행 여부
        self.b = self.add_weight(shape=(self.units), initializer='zeros',  trainable=True) # bias


    def call(self, inputs): # call 오버라이딩
        # 정의된 값들을 이용해 해당층의 로직을 수행
        return tf.matmul(inputs, self.w) + self.b # wx+b 만들어짐, 행렬곱:matmul()
    
        
class MlpModel(Model):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = Linear(2)
        self.linear2 = Linear(1)
    
    def call(self, inputs):
        # Layer의 build를 호출
        x = self.linear1(inputs)
        return self.linear2(x)
    
    
model4 = MlpModel()

opti = optimizers.Adam(learning_rate=0.1)
model4.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model4.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss metrics : ', loss_metrics)

print('설명력 : ', r2_score(y_data, model4.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 ' , model4.predict(x_data).ravel())
print(model4.summary())    
    
   


