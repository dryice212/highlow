import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os

# 데이터베이스 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
engine = create_engine(f'sqlite:///{DB_PATH}')

# 1️⃣ 데이터 로드
query = """
SELECT m.date, i.indicator_name, i.value, m.high, m.low
FROM monthly_high_low m
LEFT JOIN indicator_data i ON DATE(m.date) = DATE(i.date);
"""
df = pd.read_sql(query, engine)

print("🔹 헤드 출력")
print(df.head())  # 데이터 확인

# 2️⃣ 고점 / 저점별 지표 값 분리
df_high = df[df['high'].notna()]
df_low = df[df['low'].notna()]

# 3️⃣ 테이블 요약 (통계 분석)
stats_high = df_high.groupby("indicator_name")['value'].agg(['mean', 'std', 'min', 'max'])
stats_low = df_low.groupby("indicator_name")['value'].agg(['mean', 'std', 'min', 'max'])

# 테이블 출력
print("🔹 헤드 출력")
print(df.head())
print("🔹 고점 발생 시 지표 통계")
print(stats_high)
print("\n🔹 저점 발생 시 지표 통계")
print(stats_low)

# 4️⃣ 데이터 시각화 (히스토그램 & 박스플롯)
indicators = ["RSI_14", "MACD_line", "SMA_20", "Bollinger_upper"]  # 주요 지표 선택

plt.figure(figsize=(12, 8))
for i, indicator in enumerate(indicators):
    plt.subplot(2, 2, i+1)
    sns.histplot(df_high[df_high["indicator_name"] == indicator]["value"], color='red', label='High', kde=True)
    sns.histplot(df_low[df_low["indicator_name"] == indicator]["value"], color='blue', label='Low', kde=True)
    plt.title(f'{indicator} Distribution')
    plt.legend()
plt.tight_layout()
plt.show()

# 박스플롯 출력
plt.figure(figsize=(10, 6))
sns.boxplot(x='indicator_name', y='value', hue='high', data=df[df['indicator_name'].isin(indicators)])
plt.title("Indicator Value Distribution at Highs and Lows")
plt.xticks(rotation=45)
plt.show()
