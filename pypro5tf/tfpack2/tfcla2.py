# 로지스틱 회귀 분석 실습 소스) 2.x
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np

x_data = [[1,2],[2,3],[3,4],[4,3],[3,2],[2,1]]
y_data = [[0],[0],[0],[1],[1],[1]]

print('Sequential Api')
model = Sequential()
model.add(Dense(units=1, input_dim=2, activation='sigmoid')) # 들어오는 노드 2개, 히든레이어엔 노드가 한개-활성화함수:시그모이드, 아웃풋이 한개
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics='accuracy')
print(model.summary())

model.fit(x_data, y_data, epochs=500, batch_size=1, verbose=0)
m_eval = model.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('m_eval : ', m_eval)

# 예측
new_data = [[1,5],[10,3]]
pred = model.predict(new_data)
print('예측결과 : ', pred)
print('예측결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0))) # pred값이 0.5보다 크면1, 아니면 0
      

print('functional Api')
from keras.layers import Input
from keras.models import Model


inputs = Input(shape=(2,), batch_size=1)
outputs = Dense(units=1, activation='sigmoid')(inputs) # 이항분류라 sigmoid 다항일떈 softmax이며, 다항일땐 label에 대해 one_hot_encoding처리 해주어야함
model2 = Model(inputs, outputs)

model2.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics='accuracy')
print(model2.summary())

model2.fit(x_data, y_data, epochs=500, batch_size=1, verbose=0)
m_eval = model2.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('m_eval : ', m_eval)

# 예측
new_data = [[1,5],[10,3]]
pred = model2.predict(new_data)
print('예측결과 : ', pred)
print('예측결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0))) # pred값이 0.5보다 크면1, 아니면 0