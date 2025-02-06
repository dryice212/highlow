import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sqlalchemy import create_engine, text
from collections import deque
import random

# 데이터베이스 설정
db_path = "data/kospi200.db"
engine = create_engine(f'sqlite:///{db_path}')

# RL 결과 저장 함수
def save_training_result(episode, total_reward, loss):
    with engine.connect() as conn:
        query = text("""
            INSERT INTO rl_results (episode, total_reward, loss) 
            VALUES (:episode, :total_reward, :loss)
        """)
        conn.execute(query, {"episode": episode, "total_reward": total_reward, "loss": loss})
        conn.commit()

# 강화학습 환경 클래스
class TradingEnvironment:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.df = self.load_data()
        if self.df is None or self.df.empty:
            raise ValueError("❌ 데이터 로드 실패: 데이터가 없습니다.")
        self.current_step = 0
        self.position = 'NONE'
        self.balance = 10000.0
        self.episode_end_date = None
        self.action_space = ['LONG', 'SHORT', 'HOLD']
        self.state_size = self.df.shape[1]

    def load_data(self):
        try:
            query = text(f"""
                SELECT date, open, high, low, close, volume 
                FROM index_data 
                WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
            """)
            df = pd.read_sql(query, engine, parse_dates=['date'], index_col='date')
            if df.empty:
                print("❌ 데이터 없음: 주어진 날짜 범위에 데이터가 없습니다.")
                return None
            return df.sort_values(by='date')
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            return None

    def step(self, action_index):
        if self.df.empty or self.current_step >= len(self.df) - 1:
            return np.zeros((LOOKBACK_WINDOW, self.state_size)), 0, True

        self.current_step += 1
        action = self.action_space[action_index]
        reward = 0
        done = False

        close_price = self.df.iloc[self.current_step]['close']
        previous_close_price = self.df.iloc[self.current_step - 1]['close']
        self.episode_end_date = self.df.index[self.current_step - 1]

        if self.position == 'NONE':
            if action == 'LONG':
                self.position = 'LONG'
                self.entry_price = previous_close_price
            elif action == 'SHORT':
                self.position = 'SHORT'
                self.entry_price = previous_close_price
        elif self.position == 'LONG' and action == 'SHORT':
            reward = close_price - self.entry_price
            self.balance += reward
            self.position = 'NONE'
        elif self.position == 'SHORT' and action == 'LONG':
            reward = self.entry_price - close_price
            self.balance += reward
            self.position = 'NONE'

        return self.get_state(), reward, done

    def get_state(self):
        start_index = max(0, self.current_step - LOOKBACK_WINDOW)
        state = self.df.iloc[start_index:self.current_step, :].values
        state = np.nan_to_num(state, nan=0.0).astype(np.float32)
        return state

# 강화학습 모델
class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 학습 함수
def train_model(model, target_model, optimizer, memory, env, epsilon):
    for episode in range(NUM_EPISODES):
        state = env.get_state()
        done = False
        total_reward = 0
        losses = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action_index = random.randint(0, len(env.action_space)-1) if random.random() < epsilon else torch.argmax(q_values).item()
            next_state, reward, done = env.step(action_index)
            memory.push(state, action_index, reward, next_state, done)
            state = next_state
            total_reward += reward

        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            loss = F.mse_loss(model(torch.FloatTensor(states).to(device)), target_model(torch.FloatTensor(next_states).to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        avg_loss = np.mean(losses) if losses else 0
        save_training_result(episode + 1, total_reward, avg_loss)
        print(f"Episode {episode+1}/{NUM_EPISODES} - Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

# 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TradingEnvironment('2010-01-02', '2025-01-02')
    model = TradingModel(env.state_size, 128, 3).to(device)
    target_model = TradingModel(env.state_size, 128, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    train_model(model, target_model, optimizer, memory, env, EPSILON_START)
    print("학습 완료! RL 결과가 데이터베이스에 저장되었습니다.")
