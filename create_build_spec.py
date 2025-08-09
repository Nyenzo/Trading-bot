"""
Build Configuration for PyInstaller
Creates optimized executable for the trading bot
"""

spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

datas = [
    ('models', 'models'),
    ('historical_data', 'historical_data'),
    ('requirements.txt', '.'),
    ('README.md', '.'),
    ('EXECUTION_GUIDE.md', '.'),
]

hiddenimports = [
    'stable_baselines3',
    'stable_baselines3.common.vec_env',
    'stable_baselines3.common.callbacks',
    'gymnasium',
    'gymnasium.spaces',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.preprocessing',
    'xgboost',
    'lightgbm', 
    'catboost',
    'joblib',
    'ta',
    'ta.trend',
    'ta.momentum',
    'ta.volatility',
    'ta.volume',
    'pandas',
    'numpy',
    'streamlit',
    'plotly',
    'yfinance',
    'alpha_vantage',
    'fredapi',
    'pytz',
    'schedule'
]

a = Analysis(
    ['trading_bot_main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

ptp = Tree('models', prefix='models', excludes=["*.pyc"])
ptd = Tree('historical_data', prefix='historical_data', excludes=["*.pyc"])

a.datas += ptp
a.datas += ptd

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TradingBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Add icon if you have one
)
"""

# Write the spec file
with open("trading_bot.spec", "w") as f:
    f.write(spec_content)

print("✅ Created PyInstaller spec file: trading_bot.spec")
