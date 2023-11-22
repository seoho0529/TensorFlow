# tf.constant() : 텐서를 직접 기억
# tf.Variable() : 텐서가 저장된 주소를 참조
import tensorflow as tf
import numpy as np

node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
imsi = tf.add(node1, node2)
print(imsi)

print()
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3)
print(node4)
node4.assign_add(node3)
print(node4)

print()
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a, b)  # multiply:일반 곱,matmul:행렬 곱
result = tf.cond(a<b, lambda :tf.add(10, c),  lambda:tf.square(a))
print('result : ', result.numpy())

print('----'*10)
v = tf.Variable(1)

@tf.function # 을 쓰면 auto_Graph기능이 먹는다. -> Graph환경에서 처리가 된다. : 연산속도가 빠르기 때문에 사용
def find_next_func():
    v.assign(v+1)
    if tf.equal(v % 2, 0):  # v를 2로 나눈 나머지가 0인지
        v.assign(v+10)
        
find_next_func()
print(v.numpy())
print(type(find_next_func)) # <class 'function'>는 파이썬에서 실행됐다는 것을 의미
# <class 'tensorflow.python.eager.polymorphic_function.polymorphic_function.Function'>는 tf에서 실행된 것을 의미

print('func1 --------------------')
def func1():  # 파이썬 작업 환경에서 수행
    imsi = tf.constant(0)  # imsi=0
    su = 1
    for _ in range(3):
        imsi = tf.add(imsi, su)
    return imsi

kbs = func1()
print(kbs.numpy(), ' ', np.array(kbs))


print('func2 --------------------')
imsi = tf.constant(0)
@tf.function
def func2():  # Graph 영역 내에서 수행
    # imsi = tf.constant(0)  # imsi=0
    global imsi
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su
        imsi += su
    return imsi

kbs = func2()
print(kbs.numpy(), ' ', np.array(kbs))



print('func3 --------------------')
imsi = tf.Variable(0)
@tf.function
def func3():  # Graph 영역 내에서 수행
    #imsi = tf.Variable(0)  # auto_Graph에서 Variable()은 함수 밖에서 선언(지역변수x)
    su = 1
    for _ in range(3):
        imsi.assign_add(su)
        # imsi = imsi + su
        # imsi += su
    return imsi

kbs = func3()
print(kbs.numpy(), ' ', np.array(kbs))

print('구구단 출력 -------------')
@tf.function
def gugu1(dan):
    su = 0
    for _ in range(9):
        su = tf.add(su, 1)
        # print(su)  # good-잘 출력됨
        # print(su.numpy()) # error  - 텐서를 일반 넘파이로 하면 에러
        # auto_Graph를 사용하면 numpy, 서식있는 데이터 사용 안됨
        print('{} * {} = {}'.format(dan,su,dan*su))

gugu1(3)

print('-----'*10)
# 내장함수 : 일반적으로 numpy 지원함수를 그대로 사용 
# ... 중 reduce~ 함수
ar = [[1,2],[3,4]]
print(tf.reduce_sum(ar).numpy())
print(tf.reduce_mean(ar, axis = 0).numpy())
print(tf.reduce_mean(ar, axis = 1).numpy())


# one_hot encoding 원핫인코딩
print(tf.one_hot([0,1,2,0], depth=3))

