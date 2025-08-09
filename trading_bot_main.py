"""
Trading Bot Main Entry Point
Unified entry point for the packaged executable
"""

import sys
import os
import argparse
from pathlib import Path

def add_resource_path():
    """Add resource paths for PyInstaller bundle"""
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS)
        sys.path.insert(0, str(bundle_dir))
        os.chdir(bundle_dir)

def main():
    """Main entry point with command selection"""
    add_resource_path()
    
    parser = argparse.ArgumentParser(
        description='🤖 Hybrid ML-DRL Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trading_bot.exe trade --demo          # Demo trading
  trading_bot.exe trade --episodes 10   # Live trading  
  trading_bot.exe train                 # Train models
  trading_bot.exe dashboard             # Launch dashboard
  trading_bot.exe collect-data          # Collect market data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    trade_parser = subparsers.add_parser('trade', help='Run trading agent')
    trade_parser.add_argument('--demo', action='store_true', help='Demo mode with historical data')
    trade_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--type', choices=['ml', 'drl', 'both'], default='both', help='What to train')
    
    data_parser = subparsers.add_parser('collect-data', help='Collect market data')
    data_parser.add_argument('--historical', action='store_true', help='Download historical data')
    
    subparsers.add_parser('dashboard', help='Launch monitoring dashboard')
    
    signal_parser = subparsers.add_parser('signals', help='Generate trading signals')
    signal_parser.add_argument('--pair', default='XAUUSD', help='Currency pair')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🤖 Hybrid ML-DRL Trading Bot v1.0")
    print("=" * 50)
    
    try:
        if args.command == 'trade':
            from run_improved_hybrid_agent import run_improved_hybrid_agent
            print(f"🚀 Running trading agent (demo={args.demo}, episodes={args.episodes})")
            run_improved_hybrid_agent(demo_mode=args.demo, episodes=args.episodes)
            
        elif args.command == 'train':
            if args.type in ['ml', 'both']:
                print("🤖 Training ML ensemble models...")
                from train_evaluate import main as train_ml
                train_ml()
                
            if args.type in ['drl', 'both']:
                print("🧠 Training DRL hybrid agent...")
                from train_improved_hybrid_agent import main as train_drl
                train_drl()
                
        elif args.command == 'collect-data':
            if args.historical:
                print("📊 Downloading historical data...")
                from download_hourly_data import main as download_data
                download_data()
            else:
                print("📡 Collecting real-time data...")
                from data_collection import main as collect_data
                collect_data()
                
        elif args.command == 'dashboard':
            print("📈 Launching dashboard...")
            print("🌐 Dashboard will open in your browser at: http://localhost:8501")
            print("📝 Press Ctrl+C to stop the dashboard")
            
            # Try to run streamlit programmatically
            try:
                import subprocess
                import sys
                
                # Get the correct python executable
                python_exe = sys.executable
                dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
                
                # Run streamlit as a subprocess
                subprocess.run([python_exe, "-m", "streamlit", "run", dashboard_path, 
                              "--server.headless", "true", 
                              "--server.enableCORS", "false",
                              "--server.enableXsrfProtection", "false"])
                              
            except Exception as e:
                print(f"⚠️ Could not launch dashboard programmatically: {e}")
                print("📋 Manual command:")
                print("   streamlit run dashboard.py")
                print("   OR")
                print("   python -m streamlit run dashboard.py")
            
        elif args.command == 'signals':
            print(f"📊 Generating signals for {args.pair}...")
            from signal_predictor import main as predict_signals
            predict_signals()
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
