# 단순선형모델 작성
# 1) keras의 내장 api 사용 - Sequential : 다음번 예제
# 2) GradientTape 객체를 이용해 모델을 구현 - 유연하게 복잡한 로직을 처리할 수 있다.
# 3) TensorFlow는 GradientTape를 이용하여 쉽게 오자 역전파를 수행할 수 있다.
# 역전파를 수행하면서 w를 갱신 -> 미분을 통해 최적의 저점을 찾아내는 것 | y=wx+b의 w를 찾아내기!!


import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

tf.random.set_seed(2)
w = tf.Variable(tf.random.normal((1,))) # 정규분포를 따르는 난수를 튜플타입으로
b = tf.Variable(tf.random.normal((1,)))
print(w.numpy(), b.numpy())
opti = SGD() # 경사하강법을 쓴다는 거임 얘 대신에 RMSProp, Adam을 쓸 수도 있음

def trainModel(x, y): # train모델이 역전파를 통해 loss를 최소화하는 과정이며 이걸 짧게 요약하면 tf5의 26행임
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y))) # cost function
    grad = tape.gradient(loss, [w, b])  # 자동미분을 계산(loss를 w와 b로 미분)
    opti.apply_gradients(zip(grad, [w, b]))  # zip : 튜플 형태로 차례로 접근할 수 있는 반복자(iterator)를 반환
    return loss

x=[1.,2.,3.,4.,5.]
y=[1.2,2.0,3.0,3.5,5.5]
print(np.corrcoef(x,y))   # 0.97494708



w_val=[]
cost_val=[]

for i in range(1, 101):
    loss_val = trainModel(x,y)
    cost_val.append(loss_val.numpy())
    w_val.append(w.numpy())
    if i % 10 == 0:
        print(loss_val)
    

print('cost_val : ', cost_val)
print('w_val : ', w_val)

import matplotlib.pyplot as plt
plt.plot(w_val, cost_val, 'o')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()


print('cost가 최소일때 w:',w.numpy())
print('cost가 최소일때 b:',b.numpy())

y_pred = tf.multiply(x, w) + b  # 선형회귀식이 만들어짐 (모델이 만들어졌단거)
print('예측값 : ', y_pred.numpy())

plt.plot(x, y, 'ro', label='real y')
plt.plot(x, y_pred, 'b-', label='y_pred')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# 새 값으로 예측하기
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b
print('예측 결과 : ', new_pred.numpy())
