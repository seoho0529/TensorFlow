# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 논리회로 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam
# Dense : 노드를 구성하는 클래스

# 1. 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

# 2. 모델구성(설정)
# model = Sequential([
#     Dense(input_dim=2, units=1),
#     Activation('sigmoid')
# ])

model = Sequential()  # 순차적으로 네트워크를 쌓는다...
# model.add(Dense(units=1, input_dim=2))
# model.add(Activation('sigmoid'))
model.add(Dense(units=1, input_dim=2, activation='sigmoid')) # 이항분류일때 sigmoid, 다항분류일때 softmax

# 3. 모델 학습 과정 설정(컴파일)
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])
# 분류일때 accuracy, |  mse는 회귀에서
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# optimizer : 입력데이터와 손실함수를 업데이트하는 메커니즘이다. 손실함수의 최소값을 찾는 알고리즘을 optimizer라고 칭한다.
# learning_rate : 학습률 | momentum : 지역최소값에서 벗어나게 함
# 

# 4. 모델 학습시키기(train data) : 더 나은 표현(w-가중치 를 갱신)을 찾는 자동화 과정
model.fit(x, y, epochs=500, batch_size=1, verbose=0) 
# (학습횟수:500(적당히 주길))
# batch_size : 훈련데이터를 여러개의 작은 묶음(batch)으로 만들어 가중치(w)를 갱신. 1 epoch시 사용하는 dataset의 크기

# 5. 모델 평가(test data)
loss_metrics = model.evaluate(x, y, batch_size=1, verbose=0)
print(loss_metrics)  # [0.35304421186447144, 0.75] <-- 이 숫자들은 서로 반비례 관계

# 6. 학습결과 확인 : 예측값 출력
pred = model.predict(x, batch_size=1, verbose=0)
pred = (model.predict(x) > 0.5).astype('int32') # 0.5보다 크면 1, 작으면 0
print('예측 결과 : ', pred.flatten())


# 7. 모델 저장
model.save('tf4model.h5')  #hdf5


# 8. 모델 읽기
from keras.models import load_model
mymodel = load_model('tf4model.h5')

mypred = (mymodel.predict(x) > 0.5).astype('int32') # 0.5보다 크면 1, 작으면 0
print('예측 결과 : ', mypred.flatten())


