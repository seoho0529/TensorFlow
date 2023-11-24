# 다중선형회귀 분석
# TensorBoard : 머신러닝 실험을 위한 시각화 툴킷(toolkit)입니다. 
# TensorBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 
# 추적 및 시각화하는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는 것, 이미지를 출력하는 것 등이 가능하다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# 5명이 치른 세 번의 시험점수로 다음 번 시험 점수 예측
x_data = np.array([[70,85,80],[71,89,78],[50,85,60],[55,25,50],[50,35,10]])
y_data = np.array([73,82,72,50,34])

print('1) Sequential api ---')
model = Sequential()
model.add(Dense(units=6, input_dim=3, activation='linear', name="a")) # 독립변수3, 유닛수6, 레이어 이름은 a
model.add(Dense(units=3, activation='linear', name="b"))
model.add(Dense(units=1, activation='linear', name="c"))
print(model.summary())

opti = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=2)
print(history.history['loss'])

# plt.plot(history.history['loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

loss_metrics = model.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)  # train의 loss와 test의 loss 비교
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data))) # 설명력 :  0.9304


print('2) functional api ---')
from keras.layers import Input
from keras.models import Model

inputs =  Input(shape=(3,))
output1 = Dense(6, activation='linear', name='a')(inputs)
output2 = Dense(3, activation='linear', name='b')(output1)
output3 = Dense(1, activation='linear', name='c')(output2)
model2 = Model(inputs, output3)
print(model2.summary())

opti = optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])

# tensorboard
from keras.callbacks import TensorBoard
tb = TensorBoard(
    log_dir='./my',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

history = model2.fit(x_data, y_data, batch_size=1, epochs=50, verbose=1, callbacks=[tb])
print(history.history['loss'])

loss_metrics = model2.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))

# 새로운 값 예측
x_new = np.array([[30, 35, 30], [5, 7, 88]])













