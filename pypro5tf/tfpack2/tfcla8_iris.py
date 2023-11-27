# iris dataset 사용
# 3개의 분류모델 작성 후 성능 비교 출력 : ROC curve

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.DESCR)
print(iris.keys())
x = iris.data # feature
print(x[:2])
y = iris.target # label
print(y)
print(set(y))  # {0,1,2}
names=iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# one_hot처리 => keras : to_categorical | numpy : np.eys() | pandas : get_dummies(), sklearn : OneHotEncoder ...
onehot = OneHotEncoder(categories='auto')

print(y.shape)  # (150,)
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y.shape)  # (150, 3)
print(y[:3])

# feature : 표준화/정규화 - 일반적으로 얘를 하면 성능이 좋아진다.
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:3])

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features, ' ', n_classes)  # 4   3

# n의 개수만큼 모델 생성 함수
def create_model_func(input_dim, output_dim, out_nodes, n, model_name='model'):  # 입력값 4개 들어가서 최종값 3개란 뜻
    # print(input_dim, output_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name=model_name)
        for _ in range(n):
            model.add(Dense(units=out_nodes, input_dim=input_dim, activation='relu'))  # 은닉층
        
        model.add(Dense(units=output_dim, activation='softmax'))     # 출력층
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    return create_model  # 클로저 - 내부함수의 주소를 return

models = [create_model_func(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1,4)]

for cre_model in models:
     print()
     cre_model().summary()


print()
# 모델 성능 비교 목적을 위해..
history_dict={}
for cre_model in models:
    model = cre_model()
    print('model name : ', model.name)
    historys = model.fit(x=x_train, y=y_train, batch_size=5, epochs=50, verbose=0, validation_split=0.3)
    score = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('test loss : ', score[0])
    print('test accuracy : ', score[1])
    history_dict[model.name] = [historys, model] # history는 0번쨰, model은 1번째에 있음!!!!!!!!!

print(history_dict)  # 모델 3개중 모델3이 제일 좋은 것을 확인할 수 있음

'''
# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['acc']) # model_name의 0번째 .acc값
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation acc')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()
'''


# ROC curve
plt.figure()
plt.plot([0,1],[0,1], 'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel()) # ravel()를 사용한 차원축소
    plt.plot(fpr, tpr, label='{}, auc value:{:.3f}'.format(model_name, auc(fpr, tpr)))
plt.xlabel('fpr') # false postive label
plt.ylabel('tpr') # true
plt.legend()
plt.show()


# 판단 : 가장 좋은 모델 생성
# ...

