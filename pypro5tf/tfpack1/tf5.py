import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam
# Dense : 노드를 구성하는 클래스
# input layer -> hidden layer -> output layer

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  # XOR

model = Sequential()

# model.add(Dense(units=1, input_dim=2, activation='sigmoid'))
# model.add(units=5, input_dim=2)  # x1,x2라는 입력이 2개이고, 노드는 5개
# model.add(activation='relu') # 활성화함수 relu를 사용 : Sigmoid와 tanh가 갖는 Gradient Vanishing 문제를 해결하기 위한 함수
# Gradient Vanishing(기울기 소실) 문제는 Back Propagation에서 계산 결과와 정답과의 오차를 통해 가중치를 수정하는데,
# 입력층으로 갈수록 기울기가 작아져 가중치들이 업데이트 되지 않아 최적의 모델을 찾을 수 없는 문제가 있다.
# model.add(units=1)
# model.Add(Activation('sigmoid')) # 현재 빠져나갈땐 이항분류이기 때문에 sigmoid를 사용(다항이라면 softmax)
# 하지만 sigmoid일때 기울기 소실 문제가 발생할 우려가 있기 때문에 relu를 사용하는 것을 추천-히든 레이어안의 노드에서 relu사용

model.add(Dense(units=5, input_dim=2, activation='relu')) # 첫번째 레이어에는 노드 5개
model.add(Dense(units=1, activation='sigmoid'))  # 출력층엔 5개가 들어오고 노드 1개가 빠져나감

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x,y, epochs=100, batch_size=1, verbose=1)  # history란 변수를 두면 중간중간 기억 가능
loss_metrics = model.evaluate(x, y)
print(loss_metrics)
# 15 = (2+1)*5  |  6 = (5+1)*1

pred = (model.predict(x) > 0.5).astype('int32') # 0.5보다 크면 1, 작으면 0
print('예측 결과 : ', pred.flatten())


print(model.summary())

print()
print(model.input)
print(model.output)
print(model.weights)


print('***'*10)
print(history.history['loss']) # history를 사용해 진행과정을 볼 수 있음
print(history.history['accuracy']) # loss가 떨어지면서 accuracy가 오르는 것을 알 수 있음


# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

import pandas as pd
pd.DataFrame(history.history)['loss'].plot(figsize=(8,5))
plt.show()

