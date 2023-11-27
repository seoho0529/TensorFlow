# red & white dataset으로 이항분류 모델 만들기
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

wdf = pd.read_csv('../testdata/wine.csv', header=None)
print(wdf.head(3), wdf.shape)
print(wdf.info())
print(wdf.iloc[:, 12].unique())  # [1 0]
print(len(wdf[wdf.iloc[:, 12]==0]))  # 4898 - red
print(len(wdf[wdf.iloc[:, 12]==1]))  # 1599 - white

dataset = wdf.values
x = dataset[:,0:12]
y = dataset[:, -1]
print(x[0])
print(y[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4547, 12) (1950, 12) (4547,) (1950,)

model = Sequential()  # flatten작업이 진행되어 한줄로 진행된다?!
model.add(Dense(32, input_dim=12, activation='relu'))  # input_shape=(12,)로 input_dim을 바꿔쓸 수 있음 , 12는 독립변수개수로 생각
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

loss, acc = model.evaluate(x=x_train, y=y_train, verbose=2)
print('훈련안한 모델 정확도 : {:5.2f}%'.format(acc * 100)) # 백분률로 보기-74.86%

import os # 폴더 만들기 위해,,
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR): # MODEL_DIR가 없다면
    os.mkdir(MODEL_DIR) # 을 만들어라

fname = 'tfcla3.hdf5'
checkpoint = ModelCheckpoint(MODEL_DIR + fname, monitor='val_loss', verbose=0, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=5)
# mode: {"auto", "min", "max"}, acc-max, loss-min, 뭘쓸지 모르면 auto, patience값동안 val_loss가 줄다가 멈춤
history = model.fit(x=x_train, y=y_train, epochs=1000, batch_size=64,\
                    validation_split=0.3, callbacks=[early_stop,checkpoint]) # [조기종료,저장] 
                    # validation_split을 쓰면 loss,와 val_loss 둘을 쓸 수 있으며 loss말고 val_loss를 쓰는 것을 권장.

loss, acc = model.evaluate(x=x_test, y=y_test, verbose=2)
print('훈련한 모델 정확도 : {:5.2f}%'.format(acc * 100)) # 백분률로 보기 - 98.00%


# loss - 시각화
epoch_len = np.arange(len(history.epoch))
plt.plot(epoch_len, history.history['loss'], c='red', label='loss')
plt.plot(epoch_len, history.history['val_loss'], c='blue', label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

# accuracy - 시각화
plt.plot(epoch_len, history.history['accuracy'], c='red', label='accuracy')
plt.plot(epoch_len, history.history['val_accuracy'], c='blue', label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

print('--------------------------')
# 가장 우수한 모델로 저장된 파일을 읽어서 분류 예측
from keras.models import load_model
mymodel = load_model(MODEL_DIR + fname)
new_data=x_test[:5, :]
pred = mymodel.predict(new_data)
print('pred : ', np.where(pred > 0.5, 1, 0).flatten())
































