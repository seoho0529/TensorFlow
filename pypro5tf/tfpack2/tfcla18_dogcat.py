# CNN을 이용하여 개, 고양이 이미지 분류 (이항분류)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
import keras

data_url='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
pathto_zip = keras.utils.get_file('cats_and_dogs.zip', origin=data_url, extract=True) # extract : 압축 여부
PATH = os.path.join(os.path.dirname(pathto_zip), 'cats_and_dogs_filtered')
print(PATH)

batch_size = 128
epochs=15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# 데이터 준비
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # 고양이 사진이 들어있는 디렉토리
train_dogs_dir = os.path.join(train_dir, 'dogs')  # 개 사진이 들어있는 디렉토리
validation_cats_dir = os.path.join(validation_dir, 'cats')  # 고양이 사진이 들어있는 디렉토리
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # 개 사진이 들어있는 디렉토리

print(train_cats_dir) # C:\Users\SEOHO\.keras\datasets\cats_and_dogs_filtered\train\cats

num_cats_dir = len(os.listdir(train_cats_dir))
num_dogs_dir = len(os.listdir(train_dogs_dir))
print(os.listdir(train_cats_dir)[:5]) # ['cat.0.jpg', 'cat.1.jpg', 'cat.10.jpg', 'cat.100.jpg', 'cat.101.jpg']
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_dir + num_dogs_dir
total_val = num_cats_val + num_dogs_val

# 이미지 개수 확인
print('train_train_cat : ', num_cats_dir)  # 1000
print('train_train_dog : ', num_dogs_dir)  # 1000
print('train_validation_cat : ', num_cats_val)  # 500
print('train_validation_dog : ', num_dogs_val)  # 500
print('total_train : ', total_train)  # 2000
print('total_val : ', total_val)    # 1000










