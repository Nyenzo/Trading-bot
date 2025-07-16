import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class MultiPairTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir='historical_data', pairs=None, window_size=24):
        super().__init__()
        if pairs is None:
            pairs = ['XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.pairs = pairs
        self.data_dir = data_dir
        self.window_size = window_size
        self._load_data()
        self.n_pairs = len(self.pairs)
        self.n_features = self.data[self.pairs[0]].shape[1]
        # Observation: window_size * n_features * n_pairs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_features * self.n_pairs), dtype=np.float32)
        # Action: for each pair, 0=hold, 1=buy, 2=sell
        self.action_space = spaces.MultiDiscrete([3] * self.n_pairs)
        self.reset()

    def _load_data(self):
        self.data = {}
        min_len = np.inf
        for pair in self.pairs:
            df = pd.read_csv(os.path.join(self.data_dir, f'{pair}_hourly.csv'), index_col=0, parse_dates=True)
            df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
            df = df.fillna(method='ffill').fillna(method='bfill')
            self.data[pair] = df
            min_len = min(min_len, len(df))
        # Align all pairs to the same length
        for pair in self.pairs:
            self.data[pair] = self.data[pair].iloc[-int(min_len):]
        self.length = int(min_len)

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.positions = np.zeros(self.n_pairs, dtype=int)  # 0=flat, 1=long, -1=short
        self.entry_prices = np.zeros(self.n_pairs)
        self.total_reward = 0.0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = []
        for pair in self.pairs:
            window = self.data[pair].iloc[self.current_step - self.window_size:self.current_step].values
            obs.append(window)
        obs = np.concatenate(obs, axis=1)  # shape: (window_size, n_features * n_pairs)
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0.0
        done = False
        info = {}
        # For each pair, execute action
        for i, pair in enumerate(self.pairs):
            price = self.data[pair].iloc[self.current_step]['4. close']
            if action[i] == 1:  # Buy
                if self.positions[i] == 0:
                    self.positions[i] = 1
                    self.entry_prices[i] = price
                elif self.positions[i] == -1:
                    # Close short, open long
                    reward += self.entry_prices[i] - price
                    self.positions[i] = 1
                    self.entry_prices[i] = price
            elif action[i] == 2:  # Sell
                if self.positions[i] == 0:
                    self.positions[i] = -1
                    self.entry_prices[i] = price
                elif self.positions[i] == 1:
                    # Close long, open short
                    reward += price - self.entry_prices[i]
                    self.positions[i] = -1
                    self.entry_prices[i] = price
            else:  # Hold
                pass
        # Mark-to-market P&L for open positions
        for i, pair in enumerate(self.pairs):
            price = self.data[pair].iloc[self.current_step]['4. close']
            if self.positions[i] == 1:
                reward += price - self.entry_prices[i]
                self.entry_prices[i] = price
            elif self.positions[i] == -1:
                reward += self.entry_prices[i] - price
                self.entry_prices[i] = price
        self.total_reward += reward
        self.current_step += 1
        if self.current_step >= self.length:
            done = True
        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Positions: {self.positions}, Total Reward: {self.total_reward}") 