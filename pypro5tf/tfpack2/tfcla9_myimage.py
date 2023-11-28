# 내가 그린 숫자 이미지 모델에 분류 요청하기
from PIL import Image # 이미지 확대 축소 라이브러리
import numpy as np
import matplotlib.pyplot as plt

# 내 이미지 출력
im = Image.open('number.png')
img = np.array(im.resize((28,28), Image.LANCZOS).convert('L'))  # 이미지 크기를 28*28로 바꿈, 'L':그레이스케일, 'l':이진화 ...
# LANCZOS: 높은 해상도의 사진 또는 영상을 낮은 해상도로 변환하거나 나타낼때 깨진 패턴의 형태로 나타나게 되는데 이를 최소화 시켜주는 방법
print(img)
print(img.shape)

# plt.imshow(img, cmap='Greys')
# plt.show()

data = img.reshape([1,784])
data = data / 255.0  # 정규화
print(data)

# 학습이 끝난 모델로 내 이미지를 판별
import keras
mymodel = keras.models.load_model('tfc9model.hdf5')
pred = mymodel.predict(data)
print('pred : ', pred)
print('pred : ', np.argmax(pred, axis=1))