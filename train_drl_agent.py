import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from drl_trading_env import MultiPairTradingEnv

# Register the custom environment (if needed)
# gym.register('MultiPairTradingEnv-v0', entry_point='drl_trading_env:MultiPairTradingEnv')

env = MultiPairTradingEnv(window_size=24)
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_trading_tensorboard/"
)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("ppo_multi_pair_trading")
print("Model saved as ppo_multi_pair_trading.zip") 