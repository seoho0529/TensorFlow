# 뉴스 기사 다항 분류
# 로이터 뉴스 기사 데이터는 총 11,258개의  뉴스 기사가 46개의 뉴스 카테고리로 분류되는 뉴스 기사 데이터

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
from keras.datasets import reuters

# print(reuters.load_data(num_words=10000))
(train_data, train_label),(test_data,test_label) = reuters.load_data(num_words=10000)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
print(train_data[0])    # [1, 2, 2, 8, 43, 10, 447, 5, 25, ...
print(train_label[0])   # 3
print(set(train_label)) # {0, 1, 2, 3, ..., 41, 42, 43, 44, 45}

# 숫자로 매핑된 실제 데이터 보기 ------------------------
word_index = reuters.get_word_index()
print(word_index)
# {'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, ...
reverse_word_index = dict([value, key] for (key, value) in word_index.items())
'''
print(reverse_word_index)
print(train_data[0])
'''
decode_review = ' '.join([reverse_word_index.get(i) for i in train_data[0]])
# print(decode_review)  # 결국 단어가 아닌 숫자를 학습한 것임. 이미지의 경우도 마찬가지. 이미지를 숫자로 이루어져 있기 때문에 숫자를 학습
# [1, 2, 2, 8, 43, 10,  ==> the of of mln loss for
# ------------------------------------------------

# 데이터 
def vector_func(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):  # sequences 데이터와 색인이 만들어짐
        results[i, seq] = 1.
    return results

x_train = vector_func(train_data)  # train_data를 벡터화
x_test = vector_func(test_data)    # test_data를 벡터화

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(x_train[0])

#print(train_labels[0])
'''
# label은 원핫처리
def to_onehot_func(labels, dim=46):
    results = np.zeros((len(labels), dim))
    for i, labels in enumerate(labels):  # labels 데이터와 색인이 만들어짐
        results[i, labels] = 1.
    return results

one_hot_train_labels = to_onehot_func(train_label)
print(one_hot_train_labels[0])  # [0. 0. 0. 1. 0. 0 ...
'''

# 원핫 처리용 함수
one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_label)
print(one_hot_train_labels[0])  # [0. 0. 0. 1. 0. 0 ...

# 네트워크 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# validation data : 선택적(validation data, validation split)
x_val = x_train[:1000]  # validation으로 1000개 쓰고 그 이후것들은 train으로 사용
partial_x_train = x_train[1000:]
print(x_val.shape, partial_x_train.shape)  # (1000, 10000) (7982, 10000) -> validation 하는 이유 : 과적합 방지
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
print(y_val.shape, partial_y_train.shape)  # (1000, 46):검증, (7982, 46):학습

history = model.fit(x=partial_x_train, y=partial_y_train, epochs=20, batch_size=512,\
                     validation_data=(x_val, y_val), verbose=2)
# train, validation 중 뭐가 좋은진 돌려봐야 암

# 시각화
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

results = model.evaluate(x_test, one_hot_test_labels, batch_size=512, verbose=0)
print('results : ', results)

# 예측 pred
pred = model.predict(x_test[:3])
print(pred[0].shape)
print(np.sum(pred[0]))
print('예측값 : ', pred)
print('예측값 : ', np.argmax(pred, axis=1))
print('실제값 : ', np.argmax(one_hot_test_labels[:3], axis=1))

