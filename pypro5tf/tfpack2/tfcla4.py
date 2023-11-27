# 다항분류 : 출력값이 softmax로 인해 확률값으로 여러개가 출력(범주형으로 처리 - 가장 큰 인덱스를 취함). label은 one_hot_incoding해준다.
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


np.random.seed(1)

# dataset 작성
xdata = np.random.random((1000, 12)) # 1000행 12열 - 임의의 과목에 대한 시험점수라 가정
ydata = np.random.randint(5, size=(1000, 1)) # 시험과목은 5개 국어:0 ~ 체육:4 라고 가정

print(xdata[:2]) # feature
print(ydata[:2]) # label
ydata = to_categorical(ydata, num_classes=5)  # one_hot_incoding
print(ydata[:2]) # label
print([np.argmax(i) for i in ydata[:2]]) # label

model = Sequential()
model.add(Dense(units=32, input_shape=(12, ), activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(xdata, ydata, epochs=1000, batch_size=32, verbose=2, shuffle=True)
print('모델 평가 :', model.evaluate(xdata, ydata, batch_size=32, verbose=0))


# 시각화
plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()


# 기존 값으로 분류 예측
print('예측값 : ', model.predict(xdata[:5]))
print('예측값 : ', [np.argmax(i) for i in model.predict(xdata[:5])])
print('실제값 : ', ydata[:5])
print('실제값 : ', [np.argmax(i) for i in ydata[:5]])

print()
x_new = np.random.random([1, 12])
print(x_new)
new_pred = model.predict(x_new)
print('분류 결과 : ', new_pred)  # 확률값으로 나올것
print('분류 결과 합 : ', np.sum(new_pred))  # 1에 근사한 값으로 나올 것
print('분류 결과 : ', np.argmax(new_pred))

# 시험 과목은 5개 국어:0 ~ 체육:4라고 가정
kwamok = np.array(['국어','영어','수학','과학','체육'])
print('예측값 : ', kwamok[np.argmax(new_pred, axis=-1)])
print('실제값 : ', kwamok[np.argmax(x_new, axis=-1)])









