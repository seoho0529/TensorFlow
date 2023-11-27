# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
# * 변수 종류 *
# satisfaction_level : 직무 만족도
# last_eval‎uation : 마지막 평가점수
# number_project : 진행 프로젝트 수
# average_monthly_hours : 월평균 근무시간
# time_spend_company : 근속년수
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
# left : 이직 여부(0: 잔류, 1: 이직)
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
# sales : 부서
# salary : 임금 수준 (low, medium, high)
# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 데이터 로드
data = pd.read_csv('../testdata/HR_comma_sep.csv')

# 범주형 변수를 원핫 인코딩으로 인코딩
data_encoded = pd.get_dummies(data, columns=['sales'], drop_first=True)

# 특성 (X) 및 목표 변수 (y)로 분할
X = data_encoded.drop('salary', axis=1)
y = data_encoded['salary']

# 훈련-테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest 모델
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 예측
rf_predictions = rf_model.predict(X_test)

# 정확도
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest 정확도: {rf_accuracy}")

#Keras 모델

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

# 목표 변수 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 특성 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keras 모델
keras_model = Sequential()
keras_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
keras_model.add(Dense(32, activation='relu'))
keras_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

keras_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
keras_model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# 모델 평가
keras_loss, keras_accuracy = keras_model.evaluate(X_test_scaled, y_test_encoded)
print(f"Keras 정확도: {keras_accuracy}")

#결과 비교
print(f"Random Forest 정확도: {rf_accuracy}")
print(f"Keras 정확도: {keras_accuracy}")