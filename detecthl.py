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
SELECT DATE(i.date) AS date, i.close, 
       COALESCE(rsi.value, 0) AS RSI_14, COALESCE(macd.value, 0) AS MACD_line, 
       COALESCE(stoch.value, 0) AS Stochastic_K, COALESCE(cci.value, 0) AS CCI_20,
       COALESCE(boll_up.value, 0) AS Bollinger_upper, COALESCE(boll_low.value, 0) AS Bollinger_lower,
       COALESCE(plus_di.value, 0) AS PLUS_DI, COALESCE(minus_di.value, 0) AS MINUS_DI
FROM index_data i
LEFT JOIN indicator_data rsi ON DATE(i.date) = DATE(rsi.date) AND rsi.indicator_name = 'RSI_14'
LEFT JOIN indicator_data macd ON DATE(i.date) = DATE(macd.date) AND macd.indicator_name = 'MACD_line'
LEFT JOIN indicator_data stoch ON DATE(i.date) = DATE(stoch.date) AND stoch.indicator_name = 'Stochastic_K'
LEFT JOIN indicator_data cci ON DATE(i.date) = DATE(cci.date) AND cci.indicator_name = 'CCI_20'
LEFT JOIN indicator_data boll_up ON DATE(i.date) = DATE(boll_up.date) AND boll_up.indicator_name = 'Bollinger_upper'
LEFT JOIN indicator_data boll_low ON DATE(i.date) = DATE(boll_low.date) AND boll_low.indicator_name = 'Bollinger_lower'
LEFT JOIN indicator_data plus_di ON DATE(i.date) = DATE(plus_di.date) AND plus_di.indicator_name = 'PLUS_DI'
LEFT JOIN indicator_data minus_di ON DATE(i.date) = DATE(minus_di.date) AND minus_di.indicator_name = 'MINUS_DI'
"""
df = pd.read_sql(query, engine, parse_dates=['date'])
# 2️⃣ 고점/저점 찾기
df['High_Signal'] = (df['RSI_14'] >= 70) & (df['MACD_line'] > 0) & (df['Stochastic_K'] >= 80) & (df['CCI_20'] >= 100) & (df['close'] >= df['Bollinger_upper'])
df['Low_Signal'] = (df['RSI_14'] <= 30) & (df['MACD_line'] < 0) & (df['Stochastic_K'] <= 20) & (df['CCI_20'] <= -100) & (df['close'] <= df['Bollinger_lower'])

# 3️⃣ 고점/저점 필터링
df_highs = df[df['High_Signal']]
df_lows = df[df['Low_Signal']]

# 4️⃣ 지수 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='KOSPI 200 Index', color='blue')

# 5️⃣ 고점/저점 표시
plt.scatter(df_highs['date'], df_highs['close'], color='red', label='High Points', marker='^')
plt.scatter(df_lows['date'], df_lows['close'], color='green', label='Low Points', marker='v')

# 텍스트 추가
for _, row in df_highs.iterrows():
    plt.annotate('High', (row['date'], row['close']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')
for _, row in df_lows.iterrows():
    plt.annotate('Low', (row['date'], row['close']), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=9, color='green')

plt.xlabel('Date')
plt.ylabel('Index Value')
plt.title('KOSPI 200 Index with Detected Highs and Lows')
plt.legend()
plt.grid()
plt.show()
