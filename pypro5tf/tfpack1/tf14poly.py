# 다항회귀(Polynomial Regression) : 비선형 데이터인 경우 다항식을 사용


# 지역별 인구증가율과 고령인구비율(통계청 시각화 자료에서 발췌) 데이터로 선형회귀분석 및 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
print(len(population_inc))
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

plt.plot(population_inc,population_old,'bo')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

# 지역별 인구증가율과 고령인구비율 : 이상(극단)치 제거 - 세종시 데이터
population_inc = population_inc[:5] + population_inc[6:]  # 5번째는 제외
population_old = population_old[:5] + population_old[6:]
print(len(population_inc))

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

print('-- 다차 함수를 이용해 비선형 회귀선(2차 함수)을 작성 ---------------')
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Concatenate
from keras import optimizers, Input
import random
from keras.layers import Dense

a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())


# 잔차 제곱의 평균을 구하는 함수
def compute_loss():
    y_pred = a * population_inc * population_inc + b * population_inc + c
    loss = tf.reduce_mean((population_old - y_pred) ** 2)
    return loss

opti = optimizers.Adam(learning_rate=0.07)
for i in range(1000):
    opti.minimize(compute_loss, var_list=[a,b,c])
    
    if i % 100 == 99:
        print(i, 'a:', a.numpy(), ',b:',b.numpy, ',c:', c.numpy(), 'loss:',compute_loss().numpy())

line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x * line_x + b * line_x + c


plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc, population_old, 'bo')
plt.xlabel('지역별 인구증가율(%)')
plt.ylabel('고령 인구비율(%)')
plt.show()


print('---------------------------------')
# Sequential 사용
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(1,)),
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()


model.fit(population_inc, population_old, epochs=100)

print(model.predict(population_inc).flatten())

line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = model.predict(line_x)

# 시각화
plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc, population_old, 'bo')
plt.xlabel('지역별 인구증가율(%)')
plt.ylabel('고령 인구비율(%)')
plt.show()






