# 다중회귀분석
# feature 간 위의 차이가 큰 경우 정규화/표준화 작업이 모델의 성능을 향상

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
# 이상치가 많을때 RobustScaler 사용

data = pd.read_csv('../testdata/Advertising.csv')
print(data.head(2))
del data['no']
print(data.head(2))

fdata = data[['tv','radio','newspaper']]
ldata = data.iloc[:, [3]]
print(fdata[:2])
print(ldata[:2])

# 정규화 : (관찰값 - 최소값) / (최대값 - 최소값)
# scaler = MinMaxScaler(feature_range=(0,1))
# fdata = scaler.fit_transform(fdata)
# print(fdata[:2])

fedata = minmax_scale(fdata, axis=0, copy=True) # copy=True : 원본데이터는 보존한다는 뜻
print(fedata[:2])


# train / test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata,shuffle=True, test_size=0.3, random_state=123)
# 정규화된 fedata, 종속변수:ldata
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = Sequential() # tv  radio  newspaper라서 input_dim=3
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(10, activation='relu')) # hidden layer한개 더 추가, 20개 들어와서 10개 빠져나가기
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())


keras.utils.plot_model(model, 'tf11.png')

history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
# validation_split은 주어진 데이터에서 일정 비율(여기서는 20%)을 훈련 데이터에서 분리하여 해당 데이터를 검증 데이터로 사용
# validation_split를 사용하면 val_loss, val_mse도 볼 수 있음
# validation_data=(val_xtest, val_ytest)
loss = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print(loss)


# history
print(history.history)
print(history.history['loss'])
print(history.history['mse'])
print(history.history['val_loss'])
print(history.history['val_loss'])


# loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_test, model.predict(x_test)))


# predict
pred = model.predict(x_test[:3])
print('예측값 : ', pred.flatten())
print('실제값 : ', y_test[:3].values.flatten())


# 선형회귀분석 모델의 충족조건 : 독립성, 선형성, 정규성, 등분산성, 다중공선성 확인
