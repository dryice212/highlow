import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, Date, REAL
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# 데이터베이스 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
engine = create_engine(f'sqlite:///{DB_PATH}')
Base = declarative_base()

# 월별 저점/고점 테이블 정의
class MonthlyHighLow(Base):
    __tablename__ = 'monthly_high_low'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, unique=True)
    low = Column(REAL, nullable=True)
    high = Column(REAL, nullable=True)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# 데이터 로드
df = pd.read_sql('SELECT date, close FROM index_data', engine, parse_dates=['date'])

# 1. 로컬 최고점과 최저점 찾기
df['prev_close'] = df['close'].shift(1)
df['next_close'] = df['close'].shift(-1)

df['local_high'] = (df['prev_close'] < df['close']) & (df['next_close'] < df['close'])
df['local_low'] = (df['prev_close'] > df['close']) & (df['next_close'] > df['close'])

# 2. 7일 내 최고/최저 여부 확인 (window=3, 즉 총 7일 비교)
df['high_point'] = df['close'][(df['local_high']) & 
                               (df['close'] > df['close'].rolling(window=7, center=True).max() - 0.1)]  # 오차 허용
df['low_point'] = df['close'][(df['local_low']) & 
                              (df['close'] < df['close'].rolling(window=7, center=True).min() + 0.1)]  # 오차 허용

# 🔍 데이터 확인 (값이 있는지 체크)
print("🔹 필터링된 데이터:")
print(df[['date', 'high_point', 'low_point']].dropna().head())

# 3. 데이터 정리 후 저장
filtered_extremes = df[['date', 'high_point', 'low_point']].dropna(how='all')  # 모든 값이 NaN인 경우만 제거

for _, row in filtered_extremes.iterrows():
    if not pd.isna(row['low_point']) or not pd.isna(row['high_point']):  # 값이 하나라도 있으면 저장
        session.merge(MonthlyHighLow(date=row['date'], low=row['low_point'], high=row['high_point']))

session.commit()

# 🔍 데이터베이스 확인
count = session.query(MonthlyHighLow).count()
print(f"✅ 저장된 데이터 개수: {count}")

if count == 0:
    print("⚠ 테이블이 비어 있음. 필터링된 데이터 확인 필요.")

# 4. 📊 그래프 출력
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='KOSPI 200 Index', color='blue')

# 🔍 그래프에 찍히는지 확인
print("🟢 고점 / 저점 데이터 확인:")
print(filtered_extremes)

# 저점과 고점 화살표로 표시
plt.scatter(filtered_extremes['date'], filtered_extremes['low_point'], color='red', label='Low Points', marker='v')
plt.scatter(filtered_extremes['date'], filtered_extremes['high_point'], color='green', label='High Points', marker='^')

# 텍스트 추가
for _, row in filtered_extremes.iterrows():
    if not pd.isna(row['low_point']):
        plt.annotate('Low', (row['date'], row['low_point']), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=9, color='red')
    if not pd.isna(row['high_point']):
        plt.annotate('High', (row['date'], row['high_point']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='green')

plt.xlabel('Date')
plt.ylabel('Index Value')
plt.title('KOSPI 200 Index with Highs and Lows')
plt.legend()
plt.grid()
plt.show()
