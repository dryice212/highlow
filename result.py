import pandas as pd
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine

# 데이터베이스 설정
db_path = "data/kospi200.db"
engine = create_engine(f'sqlite:///{db_path}')

# 📌 데이터 로드 (데이터베이스에서 결과 가져오기)
def load_results():
    query = "SELECT episode, cumulative_reward, loss FROM rl_results"
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"❌ 데이터베이스에서 결과를 불러오는 중 오류 발생: {e}")
        return None

# 📌 학습 성능 시각화
def plot_results(df):
    plt.figure(figsize=(12, 5))
    
    # 보상 (Total Rewards) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(df["episode"], df["cumulative_reward"], label='cumulative_reward')
    plt.xlabel('Episodes')
    plt.ylabel('cumulative_reward')
    plt.title('Total Rewards over Episodes')
    plt.legend()
    
    # 손실 (Loss) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(df["episode"], df["loss"], label='Loss', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.legend()
    
    plt.show()

# 📌 트레이딩 성과 분석
def analyze_trading_performance(df):
    avg_reward = df["cumulative_reward"].mean()
    max_reward = df["cumulative_reward"].max()
    min_reward = df["cumulative_reward"].min()
    final_loss = df["loss"].iloc[-1]
    
    print("\n📊 트레이딩 성과 분석")
    print(f"평균 보상 (Average Reward): {avg_reward:.2f}")
    print(f"최대 보상 (Max Reward): {max_reward:.2f}")
    print(f"최소 보상 (Min Reward): {min_reward:.2f}")
    print(f"최종 손실 값 (Final Loss): {final_loss:.4f}")

# 실행
def main():
    df = load_results()
    if df is not None:
        analyze_trading_performance(df)
        plot_results(df)

if __name__ == "__main__":
    main()
