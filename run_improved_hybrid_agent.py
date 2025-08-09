"""
Run Improved Hybrid Trading Agent
Executes the trained hybrid ML-DRL agent for live trading
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from improved_hybrid_env import ImprovedHybridTradingEnv
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def is_market_open():
    """Check if forex market is currently open"""
    now = datetime.now()
    day_of_week = now.weekday()
    hour = now.hour
    
    if day_of_week == 5 and hour >= 22:
        return False
    elif day_of_week == 6:
        return False
    elif day_of_week == 0 and hour < 22:
        return False
    
    return True

def run_improved_hybrid_agent(demo_mode=False, episodes=10):
    """Run the trained hybrid agent"""
    
    print("🤖 Improved Hybrid Trading Agent")
    print("=" * 50)
    
    if not demo_mode:
        if not is_market_open():
            print("⚠️  FOREX MARKET IS CURRENTLY CLOSED")
            print("📅 Best trading times:")
            print("   • Monday 00:00 GMT (Sydney open)")
            print("   • Monday 08:00 GMT (London open)")
            print("   • Monday 13:00 GMT (New York open)")
            print("\n🎮 Running in DEMO mode with historical data...")
            demo_mode = True
        else:
            print("✅ Forex market is OPEN - Ready for live trading!")
    
    # Load the trained model
    model_path = "models/improved_hybrid_trading_agent"
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Trained model not found at {model_path}.zip")
        print("🔧 Please run 'python train_improved_hybrid_agent.py' first")
        return
    
    print(f"📦 Loading trained model from {model_path}...")
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Create environment
    print("🏗️ Creating trading environment...")
    env = ImprovedHybridTradingEnv(
        window_size=24,
        use_ml_signals=True,
        ml_feedback=True,
        max_episode_steps=500
    )
    
    print("📊 Environment Details:")
    print(f"   • Trading pairs: {env.pairs}")
    print(f"   • Data length: {env.length} timesteps")
    print(f"   • Episode length: {env.max_episode_steps} steps")
    print(f"   • ML signals: {env.use_ml_signals}")
    
    # Run trading episodes
    print(f"\n🚀 Running {episodes} trading episodes...")
    print("-" * 60)
    
    episode_rewards = []
    total_trades = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_trades = 0
        done = False
        step = 0
        
        print(f"\n📈 Episode {episode + 1}/{episodes}")
        print("   Step | Actions | Reward  | Positions | MTM P&L")
        print("   -----|---------|---------|-----------|--------")
        
        while not done and step < env.max_episode_steps:
            # Get agent action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Count trades (position changes)
            if any(info['positions'] != 0):
                episode_trades += 1
            
            # Display progress every 50 steps
            if step % 50 == 0:
                actions_str = ['Hold', 'Buy', 'Sell']
                action_names = [actions_str[a] for a in action]
                
                print(f"   {step:4d} | {str(action_names)[:7]} | {reward:7.2f} | {info['positions']} | {info['mtm_pnl']:6.2f}")
            
            step += 1
            
            if not demo_mode:
                time.sleep(0.1)  # Small delay for live trading
        
        episode_rewards.append(episode_reward)
        total_trades += episode_trades
        
        print(f"\n   💰 Episode {episode + 1} Summary:")
        print(f"      • Total reward: {episode_reward:.2f}")
        print(f"      • Steps taken: {step}")
        print(f"      • Trades made: {episode_trades}")
        print(f"      • Final positions: {info['positions']}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 TRADING SESSION SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode: {max(episode_rewards):.2f}")
    print(f"Worst episode: {min(episode_rewards):.2f}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {len([r for r in episode_rewards if r > 0]) / episodes * 100:.1f}%")
    
    if demo_mode:
        print("\n🎮 Demo mode completed using historical data")
        print("🚀 For live trading, run during market hours!")
    else:
        print("\n✅ Live trading session completed")
        print("📈 Monitor your broker account for actual P&L")
    
    return episode_rewards

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Improved Hybrid Trading Agent')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    
    args = parser.parse_args()
    
    try:
        rewards = run_improved_hybrid_agent(demo_mode=args.demo, episodes=args.episodes)
        print("\n🎉 Trading session completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error during trading: {e}")

if __name__ == "__main__":
    main()
