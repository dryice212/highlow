import os
import pandas as pd
import FinanceDataReader as fdr
from sqlalchemy import create_engine, Column, Integer, Date, REAL, TEXT, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import Index
import datetime
import talib
from sqlalchemy import text

# 데이터베이스 파일 경로 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# 데이터베이스 폴더 생성 (없을 경우)
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# 데이터베이스 연결
engine = create_engine(f'sqlite:///{DB_PATH}')

# ORM 베이스 정의
Base = declarative_base()

# 지수 데이터 테이블 모델 정의
class IndexData(Base):
    __tablename__ = 'index_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(TEXT, index=True)  # Date -> TEXT 로 변경
    open = Column(REAL)
    high = Column(REAL)
    low = Column(REAL)
    close = Column(REAL)
    volume = Column(Integer)
    change = Column(REAL)
    updown = Column(REAL)
    comp = Column(REAL)
    amount = Column(REAL)
    marcap = Column(REAL)
    highlow = Column(Integer, default=0)  # 고점/저점 컬럼 추가
    SMA_20 = Column(REAL)
    SMA_50 = Column(REAL)
    RSI_14 = Column(REAL)
    MACD_line = Column(REAL)
    MACD_signal = Column(REAL)
    Bollinger_upper = Column(REAL)
    Bollinger_middle = Column(REAL)
    Bollinger_lower = Column(REAL)
    ATR_14 = Column(REAL)
    CCI_20 = Column(REAL)
    Stochastic_K = Column(REAL)
    Stochastic_D = Column(REAL)
    ADX = Column(REAL)
    PLUS_DI = Column(REAL)
    MINUS_DI = Column(REAL)
    Ichimoku_conversion_line = Column(REAL)
    Ichimoku_base_line = Column(REAL)
    Ichimoku_leading_span_A = Column(REAL)
    Ichimoku_leading_span_B = Column(REAL)
    Ichimoku_lagging_span = Column(REAL)

# 지표 데이터 테이블 모델 정의 (더 이상 사용하지 않음)
# class IndicatorData(Base):
#     __tablename__ = 'indicator_data'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     date = Column(TEXT, index=True) # Date -> TEXT 로 변경
#     indicator_name = Column(TEXT)
#     value = Column(REAL)
#     __table_args__ = (Index('idx_date_indicator', 'date', 'indicator_name'), )

# 강화학습 결과 테이블
class RLResults(Base):
    __tablename__ = 'rl_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode = Column(Integer, index=True)
    step = Column(Integer, index=True)
    date = Column(TEXT, index=True) # Date -> TEXT 로 변경
    reward = Column(REAL)
    cumulative_reward = Column(REAL)
    action = Column(TEXT)
    state = Column(TEXT)
    next_state = Column(TEXT)
    done = Column(Integer)
    loss = Column(REAL)
    sharpe_ratio = Column(REAL)
    max_drawdown = Column(REAL)
    trade_count = Column(Integer)
    model_parameters = Column(TEXT)
    learning_rate = Column(REAL)
    epsilon = Column(REAL)
    timestamp = Column(TEXT)
    notes = Column(TEXT)

def create_database():
    """
    데이터베이스와 테이블을 생성합니다.
    """
    Base.metadata.create_all(engine)
    print(f"'{DB_PATH}'에 데이터베이스와 테이블이 생성되었습니다.")

def fetch_and_store_kospi200(start_date, end_date):
    """
    FinanceDataReader를 사용하여 KOSPI 200 데이터를 가져와 데이터베이스에 저장합니다.
    """
    try:
        # 1. KOSPI 200 데이터 가져오기 (fdr 사용)
        df = fdr.DataReader('KS200', start_date, end_date)
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Change': 'change', 'UpDown': 'updown', 'Comp':'comp', 'Amount':'amount', 'MarCap':'marcap'}, inplace=True)
        # date 컬럼을 문자열로 변환
        df['date'] = df['date'].astype(str)

        # 2. 고점/저점 계산
        df['prev_close'] = df['close'].shift(1)
        df['next_close'] = df['close'].shift(-1)

        df['local_high'] = (df['prev_close'] < df['close']) & (df['next_close'] < df['close'])
        df['local_low'] = (df['prev_close'] > df['close']) & (df['next_close'] > df['close'])

        # 3. 20일 내 최고/최저 여부 확인 (window=3, 즉 총 7일 비교)
        df['high_point'] = df['close'][(df['local_high']) &
                                       (df['close'] > df['close'].rolling(window=7, center=True).max() - 0.1)]  # 오차 허용
        df['low_point'] = df['close'][(df['local_low']) &
                                      (df['close'] < df['close'].rolling(window=7, center=True).min() + 0.1)]  # 오차 허용

        df['highlow'] = 0  # 기본값 0으로 초기화
        df.loc[df['high_point'].notna(), 'highlow'] = 1  # 고점이면 1
        df.loc[df['low_point'].notna(), 'highlow'] = -1  # 저점이면 -1

        # 4. 기술 지표 계산 (talib 라이브러리 사용)
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

        macd, macdsignal, _ = talib.MACD(df['close'])
        df['MACD_line'] = macd
        df['MACD_signal'] = macdsignal

        upperband, middleband, lowerband = talib.BBANDS(df['close'])
        df['Bollinger_upper'] = upperband
        df['Bollinger_middle'] = middleband
        df['Bollinger_lower'] = lowerband

        df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['CCI_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        df['Stochastic_K'] = stoch_k
        df['Stochastic_D'] = stoch_d

        adx = talib.ADX(df['high'], df['low'], df['close'])
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'])
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'])

        df['ADX'] = adx
        df['PLUS_DI'] = plus_di
        df['MINUS_DI'] = minus_di

        # 일목균형표 (ichimoku cloud)
        df['Ichimoku_conversion_line'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=9) # 전한선
        df['Ichimoku_base_line'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=26) # 기준선
        df['Ichimoku_leading_span_A'] = talib.SMA((df['Ichimoku_conversion_line'] + df['Ichimoku_base_line'])/2, timeperiod =26).shift(26) # 선행 스팬 1
        df['Ichimoku_leading_span_B'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=52).shift(26) # 선행 스팬 2
        df['Ichimoku_lagging_span'] = df['close'].shift(-26)  # 후행 스팬

        # 불필요한 컬럼 제거
        df.drop(columns=['prev_close', 'next_close', 'local_high', 'local_low', 'high_point', 'low_point'], inplace=True)

        # 5. 데이터베이스에 저장
        df.to_sql('index_data', engine, if_exists='replace', index=False)
        print("KOSPI 200 데이터 및 기술 지표 데이터가 'index_data' 테이블에 저장되었습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

# def calculate_and_store_indicators(start_date, end_date):
#     """
#     KOSPI 200 데이터에 기술 지표를 계산하고 데이터베이스에 저장합니다.
#     """
#     try:
#         # indicator_data 테이블 삭제 (중복 방지)
#         with engine.connect() as conn:
#             conn.execute(text("DROP TABLE IF EXISTS indicator_data"))
#         Base.metadata.tables['indicator_data'].create(engine) # 테이블 재생성
#         print("'indicator_data' 테이블을 삭제하고 다시 생성했습니다.")
#
#         df = pd.read_sql('SELECT date, open, high, low, close, volume, `change`, updown, comp, amount, marcap, highlow FROM index_data', engine)
#
#         # date 컬럼을 문자열로 변환
#         df['date'] = df['date'].astype(str)
#
#         # 지표 계산 (talib 라이브러리 사용)
#         df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
#         df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
#         df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
#
#         macd, macdsignal, _ = talib.MACD(df['close'])
#         df['MACD_line'] = macd
#         df['MACD_signal'] = macdsignal
#
#         upperband, middleband, lowerband = talib.BBANDS(df['close'])
#         df['Bollinger_upper'] = upperband
#         df['Bollinger_middle'] = middleband
#         df['Bollinger_lower'] = lowerband
#
#         df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
#         df['CCI_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
#
#         stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
#         df['Stochastic_K'] = stoch_k
#         df['Stochastic_D'] = stoch_d
#
#         adx = talib.ADX(df['high'], df['low'], df['close'])
#         plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'])
#         minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'])
#
#         df['ADX'] = adx
#         df['PLUS_DI'] = plus_di
#         df['MINUS_DI'] = minus_di
#
#         # 일목균형표 (ichimoku cloud)
#         df['Ichimoku_conversion_line'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=9) # 전한선
#         df['Ichimoku_base_line'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=26) # 기준선
#         df['Ichimoku_leading_span_A'] = talib.SMA((df['Ichimoku_conversion_line'] + df['Ichimoku_base_line'])/2, timeperiod =26).shift(26) # 선행 스팬 1
#         df['Ichimoku_leading_span_B'] = talib.SMA((df['high'] + df['low']) / 2, timeperiod=52).shift(26) # 선행 스팬 2
#         df['Ichimoku_lagging_span'] = df['close'].shift(-26)  # 후행 스팬
#
#         # 데이터베이스 저장
#         indicator_list = [
#             'SMA_20', 'SMA_50', 'RSI_14', 'MACD_line', 'MACD_signal',
#             'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower',
#             'ATR_14', 'CCI_20', 'Stochastic_K', 'Stochastic_D',
#              'ADX', 'PLUS_DI', 'MINUS_DI',
#             'Ichimoku_conversion_line', 'Ichimoku_base_line', 'Ichimoku_leading_span_A',
#             'Ichimoku_leading_span_B', 'Ichimoku_lagging_span'
#         ] # 넣을 지표 추가
#         for indicator in indicator_list:
#             indicator_df = pd.DataFrame({
#                'date': df['date'],
#                'indicator_name': indicator,
#                'value': df[indicator]
#             })
#             indicator_df['date'] = indicator_df['date'].astype(str)
#             indicator_df.to_sql('indicator_data', engine, if_exists='append', index=False)
#         print("기술 지표 데이터가 'indicator_data' 테이블에 저장되었습니다.")
#     except Exception as e:
#         print(f"에러 발생: {e}")

if __name__ == '__main__':
    start_date = '2010-01-02'
    end_date = '2025-01-02'
    create_database()
    fetch_and_store_kospi200(start_date, end_date)
    # calculate_and_store_indicators(start_date, end_date)
