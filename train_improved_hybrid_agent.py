"""
Training script for the improved hybrid environment with better episode management
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from improved_hybrid_env import ImprovedHybridTradingEnv


def train_improved_hybrid_agent(
    total_timesteps=100000,
    model_output="models/improved_hybrid_trading_agent",
    best_model_dir="./models/",
    log_dir="./logs/",
    test_episodes=5,
):
    """Train hybrid ML-DRL agent with improved episode management"""

    print("🚀 Training Improved Hybrid ML-DRL Agent")
    print("Key Improvements:")
    print("  • Episode truncation at 500 steps (vs 11,500)")
    print("  • Random episode starting points")
    print("  • Early termination on large P&L")
    print("  • Better reward shaping")
    print("=" * 60)

    print("🏗️ Creating improved hybrid trading environment...")
    env = ImprovedHybridTradingEnv(
        window_size=24, use_ml_signals=True, ml_feedback=True, max_episode_steps=500
    )

    env = DummyVecEnv([lambda: env])

    print("✅ Environment created successfully")
    print("📊 Environment Details:")
    print(f"   • Observation space: {env.observation_space}")
    print(f"   • Action space: {env.action_space}")
    print(f"   • Max episode steps: 500 (vs previous 11,500)")
    print(f"   • Trading pairs: {env.get_attr('pairs')[0]}")

    print("🧠 Creating PPO model with optimized hyperparameters...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./improved_hybrid_tensorboard/",
        device="cpu",
    )

    print("✅ PPO model created")

    # Set up evaluation
    eval_env = DummyVecEnv(
        [
            lambda: ImprovedHybridTradingEnv(
                window_size=24,
                use_ml_signals=True,
                ml_feedback=True,
                max_episode_steps=500,
            )
        ]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=2000,  # More frequent evaluation
        deterministic=True,
        render=False,
    )

    # Train the model
    print("🚀 Starting training with improved episode management...")
    print("Expected improvements:")
    print("  • Faster convergence due to shorter episodes")
    print("  • Better exploration with random starting points")
    print("  • More stable learning with frequent feedback")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="ImprovedHybridPPO",
        )

        print("✅ Training completed successfully!")

        # Save the final model
        model.save(model_output)
        print(f"💾 Model saved as {model_output}")

        # Test the trained agent
        print("🧪 Testing trained agent...")
        obs = env.reset()
        total_reward = 0
        episode_rewards = []

        for episode in range(test_episodes):
            episode_reward = 0
            obs = env.reset()

            for step in range(500):  # Fixed episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                if done[0]:
                    break

            episode_rewards.append(episode_reward)
            print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}")

        avg_reward = np.mean(episode_rewards)
        print(f"📊 Test completed. Average reward: {avg_reward:.2f}")

        return model

    except Exception as e:
        print(f"❌ Training error: {e}")
        return None


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train improved hybrid PPO agent")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--model-output", default="models/improved_hybrid_trading_agent")
    parser.add_argument("--best-model-dir", default="./models/")
    parser.add_argument("--log-dir", default="./logs/")
    parser.add_argument("--test-episodes", type=int, default=5)
    args = parser.parse_args()
    print("🤖 Improved Hybrid ML-DRL Trading System")
    print("Addressing long episode problems with:")
    print("  • Episode truncation")
    print("  • Random starting points")
    print("  • Early termination")
    print("  • Better reward shaping")
    print("=" * 60)

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Train the agent
    model = train_improved_hybrid_agent(
        total_timesteps=args.timesteps,
        model_output=args.model_output,
        best_model_dir=args.best_model_dir,
        log_dir=args.log_dir,
        test_episodes=args.test_episodes,
    )

    if model:
        print("🎉 Training completed successfully!")
        print("🚀 The improved agent should learn much faster!")
        print("📈 Episode rewards should converge more quickly")
    else:
        print("❌ Training failed. Please check the logs.")


if __name__ == "__main__":
    main()
