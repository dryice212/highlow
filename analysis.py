import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1️⃣ 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    print("✅ 데이터 로드 완료. 데이터 크기:", df.shape)
    print(df.head())
    return df

# 2️⃣ 지표를 구간별로 나누기
def categorize_indicators(df):
    df['RSI_bin'] = pd.cut(df['RSI_14'], bins=[0, 30, 50, 70, 90, 100], labels=['low', 'mid-low', 'mid', 'high', 'very-high'])
    df['MACD_bin'] = pd.cut(df['MACD_line'], bins=[-np.inf, 0, 2, np.inf], labels=['negative', 'low', 'high'])
    df['Stochastic_bin'] = pd.cut(df['Stochastic_K'], bins=[0, 20, 50, 80, 100], labels=['low', 'mid-low', 'mid-high', 'high'])
    df['CCI_bin'] = pd.cut(df['CCI_20'], bins=[-np.inf, -100, 0, 100, np.inf], labels=['very-low', 'low', 'mid', 'high'])
    print("✅ 지표 구간화 완료")
    return df

# 3️⃣ 빈도 분석 (각 구간에서 다른 지표 값의 빈도 체크)
def analyze_frequencies(df):
    freq_table = df[['RSI_bin', 'MACD_bin', 'Stochastic_bin', 'CCI_bin']].value_counts().reset_index()
    freq_table.columns = ['RSI_bin', 'MACD_bin', 'Stochastic_bin', 'CCI_bin', 'count']
    print("✅ 빈도 분석 완료")
    return freq_table

# 4️⃣ K-Means 클러스터링 적용
def apply_kmeans(df, n_clusters=3):
    numeric_cols = ['RSI_14', 'MACD_line', 'Stochastic_K', 'CCI_20']
    df_filtered = df[numeric_cols].dropna()
    
    if df_filtered.shape[0] < n_clusters:
        print("⚠️ 클러스터링을 수행하기에 데이터가 부족합니다. K-Means를 건너뜁니다.")
        df['Cluster'] = -1  # 클러스터링 실패 시 -1 할당
        return df, None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[df_filtered.index, 'Cluster'] = kmeans.fit_predict(df_filtered)
    print("✅ K-Means 클러스터링 완료")
    return df, kmeans

# 5️⃣ 실행 함수
def main(file_path):
    df = load_data(file_path)
    df = categorize_indicators(df)
    freq_table = analyze_frequencies(df)
    df, kmeans = apply_kmeans(df)
    
    print("🔹 빈도 분석 결과:")
    print(freq_table.head())
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='RSI_bin', hue='MACD_bin')
    plt.title('RSI vs MACD 빈도 분포')
    plt.xticks(rotation=45)
    plt.show()
    
    return df, freq_table, kmeans

# 예제 실행
file_path = "D:/myenv/KOSPI_ML_Trading/data/high_indicators.csv"
df_result, freq_table, kmeans_model = main(file_path)
