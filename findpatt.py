import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Date, REAL
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# 데이터베이스 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
engine = create_engine(f'sqlite:///{DB_PATH}')
Base = declarative_base()

# 고점/저점 지표 테이블 정의
class HighIndicators(Base):
    __tablename__ = 'high_indicators'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, unique=True)

class LowIndicators(Base):
    __tablename__ = 'low_indicators'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, unique=True)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# 1️⃣ 데이터 로드 (고점/저점 날짜와 지표 값 조인)
query = """
SELECT m.date, i.indicator_name, i.value, m.high, m.low
FROM monthly_high_low m
LEFT JOIN indicator_data i ON DATE(m.date) = DATE(i.date)
"""
df = pd.read_sql(query, engine)

# 2️⃣ 고점/저점 데이터 분리
df_high = df[df['high'].notna()]
df_low = df[df['low'].notna()]

# 3️⃣ 피벗 테이블로 변환 (날짜 기준으로 모든 지표 값을 컬럼으로 변환)
df_high_pivot = df_high.pivot(index='date', columns='indicator_name', values='value').reset_index()
df_low_pivot = df_low.pivot(index='date', columns='indicator_name', values='value').reset_index()

# 4️⃣ 데이터베이스에 저장
df_high_pivot.to_sql('high_indicators', engine, if_exists='replace', index=False)
df_low_pivot.to_sql('low_indicators', engine, if_exists='replace', index=False)

# 5️⃣ 테이블 저장 (CSV로 저장 가능)
df_high_pivot.to_csv(os.path.join(DB_FOLDER, 'high_indicators.csv'), index=False)
df_low_pivot.to_csv(os.path.join(DB_FOLDER, 'low_indicators.csv'), index=False)

# 결과 출력
print("🔹 고점일 때의 지표 데이터 테이블 (데이터베이스 및 high_indicators.csv 저장됨)")
print(df_high_pivot.head())
print("\n🔹 저점일 때의 지표 데이터 테이블 (데이터베이스 및 low_indicators.csv 저장됨)")
print(df_low_pivot.head())
