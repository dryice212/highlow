import os
import pandas as pd
import FinanceDataReader as fdr
from sqlalchemy import create_engine, Column, Integer, Date, REAL, TEXT, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import Index
import datetime
import talib

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
    date = Column(Date, index=True)
    open = Column(REAL)
    high = Column(REAL)
    low = Column(REAL)
    close = Column(REAL)
    volume = Column(Integer)
    change = Column(REAL)
    updown = Column(Integer)
    comp = Column(REAL)
    amount = Column(REAL)
    marcap = Column(REAL)

# 지표 데이터 테이블 모델 정의
class IndicatorData(Base):
    __tablename__ = 'indicator_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    indicator_name = Column(TEXT)
    value = Column(REAL)
    __table_args__ = (Index('idx_date_indicator', 'date', 'indicator_name'), )

# 강화학습 결과 테이블
class RLResults(Base):
    __tablename__ = 'rl_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode = Column(Integer, index=True)
    step = Column(Integer, index=True)
    date = Column(Date, index=True)
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
        df['date'] = pd.to_datetime(df['date'])

        # 2. 데이터베이스에 저장
        df.to_sql('index_data', engine, if_exists='replace', index=False)
        print("KOSPI 200 데이터가 'index_data' 테이블에 저장되었습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

def calculate_and_store_indicators(start_date, end_date):
    """
    KOSPI 200 데이터에 기술 지표를 계산하고 데이터베이스에 저장합니다.
    """
    try:
        df = pd.read_sql('SELECT date, open, high, low, close, volume, `change`, updown, comp, amount, marcap FROM index_data', engine, parse_dates=['date'], index_col='date')

        # 지표 계산 (talib 라이브러리 사용)
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

        # 데이터베이스 저장
        indicator_list = [
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD_line', 'MACD_signal',
            'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower',
            'ATR_14', 'CCI_20', 'Stochastic_K', 'Stochastic_D',
             'ADX', 'PLUS_DI', 'MINUS_DI',
            'Ichimoku_conversion_line', 'Ichimoku_base_line', 'Ichimoku_leading_span_A',
            'Ichimoku_leading_span_B', 'Ichimoku_lagging_span'
        ] # 넣을 지표 추가
        for indicator in indicator_list:
            indicator_df = pd.DataFrame({
               'date': df.index,
               'indicator_name': indicator,
               'value': df[indicator]
            })
            indicator_df.to_sql('indicator_data', engine, if_exists='append', index=False)
        print("기술 지표 데이터가 'indicator_data' 테이블에 저장되었습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    start_date = '2010-01-02'
    end_date = '2025-01-02'
    create_database()
    fetch_and_store_kospi200(start_date, end_date)
    calculate_and_store_indicators(start_date, end_date)

    # 테이블 확인 코드
    # from sqlalchemy import MetaData, create_engine
    # metadata = MetaData()
    # metadata.reflect(bind=engine)
    # print("\n테이블 목록:")
    # for table in metadata.tables.keys():
    #     print(table)
    # print("\nindex_data 테이블 데이터 확인:")
    # print(pd.read_sql_table('index_data', engine))
    # print("\nindicator_data 테이블 데이터 확인:")
    # print(pd.read_sql_table('indicator_data', engine))