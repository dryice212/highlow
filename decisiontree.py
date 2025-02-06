import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ 데이터 로드 및 전처리
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.drop(columns=['date'])  # 날짜 컬럼 제거 (모델 학습에 필요 없음)
    df = df.dropna()  # 결측치 제거
    return df

# 2️⃣ 고점 여부를 이진 분류 (1: 고점, 0: 비고점)
def create_labels(df):
    df['High_Label'] = 1  # 현재 데이터는 전부 고점이므로, 모두 1로 설정
    return df

# 3️⃣ 의사결정 트리 학습 및 시각화
def train_decision_tree(df):
    X = df.drop(columns=['High_Label'])  # 특징값
    y = df['High_Label']  # 레이블
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 정확도 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'✅ 모델 정확도: {accuracy:.2f}')
    
    # 트리 시각화
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=['Not High', 'High'], filled=True)
    plt.show()
    
    return model

# 4️⃣ 실행 함수
def main(file_path):
    df = load_data(file_path)
    df = create_labels(df)
    model = train_decision_tree(df)
    return model

# 예제 실행
file_path = "D:/myenv/KOSPI_ML_Trading/data/high_indicators.csv"
dt_model = main(file_path)
