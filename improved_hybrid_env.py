"""
Improved Hybrid Trading Environment with Episode Management
Fast learning environment with optimized episode length
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import ta
import joblib
import os
from typing import Dict, List, Optional, Tuple

class ImprovedHybridTradingEnv(gym.Env):
    """Improved trading environment with better episode management"""
    
    def __init__(self, window_size: int = 24, data_dir: str = "historical_data", 
                 use_ml_signals: bool = True, ml_feedback: bool = True,
                 max_episode_steps: int = 500):
        super().__init__()
        
        self.window_size = window_size
        self.data_dir = data_dir
        self.use_ml_signals = use_ml_signals
        self.ml_feedback = ml_feedback
        self.max_episode_steps = max_episode_steps
        
        self.pairs = ['XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        self.ml_models = {}
        self._load_ml_models()
        
        self._load_data()
        
        self.n_pairs = len(self.pairs)
        sample_data = self._add_technical_indicators(self.data[self.pairs[0]].copy())
        self.n_features = len(sample_data.columns)
        
        print(f"🔍 Features per pair: {self.n_features}")
        
        base_features = self.n_features * self.n_pairs
        ml_features = 4 * self.n_pairs if self.use_ml_signals else 0
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, base_features + ml_features), 
            dtype=np.float32)
        
        # Action space: for each pair, 0=hold, 1=buy, 2=sell
        self.action_space = spaces.MultiDiscrete([3] * self.n_pairs)
        
        # Episode tracking
        self.episode_step = 0
        self.episode_start_step = self.window_size
        
        # Trading state
        self.current_step = self.window_size
        self.positions = np.zeros(self.n_pairs)
        self.entry_prices = np.zeros(self.n_pairs)
        self.step_count = 0
        self.episode_pnl = 0.0
        
        # Performance tracking
        self.ml_performance = {pair: {'correct': 0, 'total': 0} for pair in self.pairs}
        
        print(f"✅ Improved Hybrid Environment initialized")
        print(f"   • Max episode steps: {self.max_episode_steps}")
        print(f"   • Observation space: {self.observation_space.shape}")
        print(f"   • Action space: {self.action_space}")

    def _load_ml_models(self):
        """Load pre-trained ML models"""
        models_dir = "models"
        for pair in self.pairs:
            model_path = os.path.join(models_dir, f"{pair}_model.pkl")
            if os.path.exists(model_path):
                try:
                    self.ml_models[pair] = joblib.load(model_path)
                    print(f"✅ Loaded ML model for {pair}")
                except Exception as e:
                    print(f"⚠️ Error loading ML model for {pair}: {e}")
                    self.ml_models[pair] = None
            else:
                print(f"⚠️ No ML model found for {pair}")
                self.ml_models[pair] = None

    def _load_data(self):
        """Load historical market data"""
        self.data = {}
        min_len = np.inf
        
        for pair in self.pairs:
            df = pd.read_csv(os.path.join(self.data_dir, f'{pair}_hourly.csv'), 
                           index_col=0, parse_dates=True)
            
            # Keep essential columns
            essential_cols = ['1. open', '2. high', '3. low', '4. close', '5. volume']
            df = df[[col for col in essential_cols if col in df.columns]]
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Clean data
            df = df.ffill().bfill()
            df = df.dropna()
            
            self.data[pair] = df
            min_len = min(min_len, len(df))
        
        # Align all pairs to the same length
        for pair in self.pairs:
            self.data[pair] = self.data[pair].iloc[-int(min_len):]
        
        self.length = int(min_len)
        print(f"📊 Loaded data: {self.length} timesteps for {len(self.pairs)} pairs")

    def _add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        if len(df) < 50:
            return df
            
        try:
            # Basic indicators
            df['SMA_10'] = ta.trend.sma_indicator(df['4. close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['4. close'], window=20)
            df['EMA_10'] = ta.trend.ema_indicator(df['4. close'], window=10)
            df['EMA_20'] = ta.trend.ema_indicator(df['4. close'], window=20)
            df['RSI'] = ta.momentum.rsi(df['4. close'], window=14)
            df['Williams_%R'] = ta.momentum.williams_r(df['2. high'], df['3. low'], df['4. close'], lbp=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['4. close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            
            # MACD
            macd = ta.trend.MACD(df['4. close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # ATR
            atr = ta.volatility.AverageTrueRange(high=df['2. high'], low=df['3. low'], 
                                               close=df['4. close'], window=14)
            df['ATR'] = atr.average_true_range()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high=df['2. high'], low=df['3. low'], 
                                                   close=df['4. close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # ADX
            adx = ta.trend.ADXIndicator(high=df['2. high'], low=df['3. low'], 
                                      close=df['4. close'], window=14)
            df['ADX'] = adx.adx()
            
            # Volume and other indicators
            df['OBV'] = ta.volume.on_balance_volume(df['4. close'], df['5. volume'])
            cci = ta.trend.CCIIndicator(high=df['2. high'], low=df['3. low'], 
                                      close=df['4. close'], window=20)
            df['CCI'] = cci.cci()
            df['vix'] = 20.0  # Default VIX
            
        except Exception as e:
            print(f"⚠️ Error adding technical indicators: {e}")
        
        return df

    def _get_ml_signal(self, pair: str, current_step: int) -> Tuple[int, float, float, float]:
        """Generate ML signal on-demand for a specific pair and timestep"""
        if not self.use_ml_signals or self.ml_models[pair] is None:
            return 0, 0.5, 0.5, 0.0
        
        try:
            if current_step < 50:
                return 0, 0.5, 0.5, 0.0
            
            # Simplified feature extraction for speed
            recent_data = self.data[pair].iloc[current_step-5:current_step+1]
            
            # Use only basic features to avoid mismatch
            basic_features = [
                recent_data['RSI'].iloc[-1] if 'RSI' in recent_data.columns else 50.0,
                recent_data['MACD'].iloc[-1] if 'MACD' in recent_data.columns else 0.0,
                recent_data['ATR'].iloc[-1] if 'ATR' in recent_data.columns else 0.01,
                recent_data['SMA_10'].iloc[-1] if 'SMA_10' in recent_data.columns else recent_data['4. close'].iloc[-1],
                recent_data['SMA_20'].iloc[-1] if 'SMA_20' in recent_data.columns else recent_data['4. close'].iloc[-1],
            ]
            
            # Pad to expected size (43 features)
            features = basic_features + [0.0] * (43 - len(basic_features))
            features = np.array(features[:43])  # Ensure exactly 43 features
            
            # Get prediction
            pred_proba = self.ml_models[pair].predict_proba([features])[0]
            prediction = self.ml_models[pair].predict([features])[0]
            
            # Convert to trading signal
            if prediction == 1:
                signal = 1  # Buy
            elif prediction == 2:
                signal = 2  # Sell
            else:
                signal = 0  # Hold
            
            confidence = max(pred_proba)
            
            # Get accuracy from historical performance
            if self.ml_performance[pair]['total'] > 0:
                accuracy = self.ml_performance[pair]['correct'] / self.ml_performance[pair]['total']
            else:
                accuracy = 0.5
            
            return signal, confidence, accuracy, 0.0
            
        except Exception as e:
            # print(f"⚠️ Error generating ML signal for {pair}: {e}")
            return 0, 0.5, 0.5, 0.0

    def _get_observation(self):
        """Get current observation with on-demand ML signals"""
        # Get market data window for all pairs
        market_obs = []
        
        for pair in self.pairs:
            window_data = self.data[pair].iloc[
                self.current_step - self.window_size:self.current_step
            ].values
            market_obs.append(window_data)
        
        # Concatenate market data
        obs = np.concatenate(market_obs, axis=1)
        
        # Add ML signals if enabled
        if self.use_ml_signals:
            ml_signals = []
            
            for pair in self.pairs:
                signal, confidence, accuracy, feedback = self._get_ml_signal(pair, self.current_step)
                ml_signals.extend([signal, confidence, accuracy, feedback])
            
            # Add ML signals as additional features to each timestep
            ml_array = np.tile(ml_signals, (self.window_size, 1))
            obs = np.concatenate([obs, ml_array], axis=1)
        
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment with random starting point"""
        super().reset(seed=seed)
        
        # Choose random starting point in data (not too close to end)
        max_start = self.length - self.max_episode_steps - self.window_size - 100
        if max_start < self.window_size:
            max_start = self.window_size
        
        self.episode_start_step = np.random.randint(self.window_size, max_start)
        self.current_step = self.episode_start_step
        
        # Reset trading state
        self.positions = np.zeros(self.n_pairs)
        self.entry_prices = np.zeros(self.n_pairs)
        self.step_count = 0
        self.episode_step = 0
        self.episode_pnl = 0.0
        
        return self._get_observation(), {}

    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1
        self.episode_step += 1
        
        reward = 0.0
        
        # Process actions for each pair
        for i, pair in enumerate(self.pairs):
            current_price = self.data[pair].iloc[self.current_step]['4. close']
            
            # Calculate reward for this action
            pair_reward = 0.0
            
            if action[i] == 1:  # Buy
                if self.positions[i] == 0:  # Open long
                    self.positions[i] = 1
                    self.entry_prices[i] = current_price
                elif self.positions[i] == -1:  # Close short, open long
                    pair_reward += (self.entry_prices[i] - current_price) / self.entry_prices[i] * 1000
                    self.positions[i] = 1
                    self.entry_prices[i] = current_price
                    
            elif action[i] == 2:  # Sell
                if self.positions[i] == 0:  # Open short
                    self.positions[i] = -1
                    self.entry_prices[i] = current_price
                elif self.positions[i] == 1:  # Close long, open short
                    pair_reward += (current_price - self.entry_prices[i]) / self.entry_prices[i] * 1000
                    self.positions[i] = -1
                    self.entry_prices[i] = current_price
            
            reward += pair_reward
        
        # Mark-to-market P&L for open positions
        mtm_pnl = 0.0
        for i, pair in enumerate(self.pairs):
            if self.positions[i] != 0:
                current_price = self.data[pair].iloc[self.current_step]['4. close']
                if self.positions[i] == 1:  # Long position
                    mtm_pnl += (current_price - self.entry_prices[i]) / self.entry_prices[i] * 100
                else:  # Short position
                    mtm_pnl += (self.entry_prices[i] - current_price) / self.entry_prices[i] * 100
        
        reward += mtm_pnl * 0.01  # Small weight for unrealized P&L
        self.episode_pnl += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check for episode termination
        done = False
        truncated = False
        
        if self.current_step >= self.length - 1:
            done = True  # Reached end of data
        elif self.episode_step >= self.max_episode_steps:
            truncated = True  # Episode length limit reached
        elif abs(self.episode_pnl) > 2000:  # Large loss/gain termination
            done = True
        
        # Info
        info = {
            'positions': self.positions.copy(),
            'step': self.step_count,
            'episode_step': self.episode_step,
            'mtm_pnl': mtm_pnl,
            'episode_pnl': self.episode_pnl,
            'is_success': self.episode_pnl > 50  # Define success criteria
        }
        
        return self._get_observation(), reward, done, truncated, info

    def render(self, mode='human'):
        """Render the environment"""
        print(f"Episode Step: {self.episode_step}/{self.max_episode_steps}, "
              f"Positions: {self.positions}, Episode P&L: {self.episode_pnl:.2f}")
