"""
Trading Bot Icon Creator
Creates a professional icon for the trading bot executable
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_trading_bot_icon(size=256):
    """Create a modern trading bot icon"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    primary_color = '#1E3A8A'
    secondary_color = '#F59E0B'
    accent_color = '#10B981'
    red_color = '#EF4444'
    bg_color = '#F8FAFC'
    
    candlestick_positions = [1.5, 2.2, 2.9, 3.6, 4.3, 5.0, 5.7, 6.4, 7.1, 7.8, 8.5]
    candlestick_data = [
        (1.2, 0.8, 1.0, 0.9, True),
        (1.5, 1.0, 1.1, 1.4, True),
        (1.6, 1.2, 1.4, 1.3, False),
        (1.8, 1.1, 1.3, 1.7, True),
        (2.0, 1.5, 1.7, 1.6, False),
        (2.2, 1.4, 1.6, 2.0, True),
        (2.1, 1.8, 2.0, 1.9, False),
        (2.3, 1.7, 1.9, 2.2, True),
        (2.4, 2.0, 2.2, 2.1, False),
        (2.6, 1.9, 2.1, 2.5, True),
        (2.7, 2.3, 2.5, 2.4, False)
    ]
    
    for i, (pos, (high, low, open_price, close, bullish)) in enumerate(zip(candlestick_positions, candlestick_data)):
        color = accent_color if bullish else red_color
        
        # Scale and position
        base_y = 1.5
        high *= 0.8
        low *= 0.8
        open_price *= 0.8
        close *= 0.8
        
        # Wick (thin line)
        ax.plot([pos, pos], [base_y + low, base_y + high], 
                color=color, linewidth=1, alpha=0.3)
        
        # Body (rectangle)
        body_height = abs(close - open_price)
        body_bottom = base_y + min(open_price, close)
        
        body = patches.Rectangle((pos-0.08, body_bottom), 0.16, body_height,
                               facecolor=color, edgecolor=color, alpha=0.4)
        ax.add_patch(body)
    
    # Main robot container (rounded square)
    main_container = FancyBboxPatch((2.5, 4), 5, 4.5,
                                   boxstyle="round,pad=0.2",
                                   facecolor=bg_color,
                                   edgecolor=primary_color,
                                   linewidth=3,
                                   alpha=0.95)
    ax.add_patch(main_container)
    
    # Robot head (more geometric/robotic)
    head = FancyBboxPatch((3.5, 6.5), 3, 1.8,
                         boxstyle="round,pad=0.05",
                         facecolor=primary_color,
                         edgecolor='none',
                         alpha=0.9)
    ax.add_patch(head)
    
    # Robot eyes (more digital/LED-like)
    left_eye = patches.Rectangle((4.0, 7.1), 0.4, 0.4,
                               facecolor=secondary_color, edgecolor='none')
    right_eye = patches.Rectangle((5.6, 7.1), 0.4, 0.4,
                                facecolor=secondary_color, edgecolor='none')
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    # LED indicator dots in eyes
    left_led = Circle((4.2, 7.3), 0.08, facecolor='white', edgecolor='none')
    right_led = Circle((5.8, 7.3), 0.08, facecolor='white', edgecolor='none')
    ax.add_patch(left_led)
    ax.add_patch(right_led)
    
    # Robot mouth (digital display style)
    mouth_rect = patches.Rectangle((4.3, 6.7), 1.4, 0.2,
                                 facecolor=secondary_color, edgecolor='none', alpha=0.8)
    ax.add_patch(mouth_rect)
    
    # Body panel with screen
    screen = FancyBboxPatch((3.8, 4.8), 2.4, 1.4,
                           boxstyle="round,pad=0.05",
                           facecolor='black',
                           edgecolor=accent_color,
                           linewidth=2,
                           alpha=0.8)
    ax.add_patch(screen)
    
    # Mini chart on screen
    chart_x = np.linspace(4.0, 6.0, 20)
    chart_y = 5.5 + 0.15 * np.sin(chart_x * 4) + 0.05 * (chart_x - 4.0)
    ax.plot(chart_x, chart_y, color=accent_color, linewidth=2)
    
    # Control buttons
    button1 = Circle((3.2, 5.5), 0.15, facecolor=accent_color, edgecolor='none', alpha=0.7)
    button2 = Circle((3.2, 5.0), 0.15, facecolor=red_color, edgecolor='none', alpha=0.7)
    button3 = Circle((6.8, 5.5), 0.15, facecolor=secondary_color, edgecolor='none', alpha=0.7)
    button4 = Circle((6.8, 5.0), 0.15, facecolor=primary_color, edgecolor='none', alpha=0.7)
    
    ax.add_patch(button1)
    ax.add_patch(button2)
    ax.add_patch(button3)
    ax.add_patch(button4)
    
    # Antenna/signal indicators
    ax.plot([4.5, 4.3], [8.3, 8.8], color=secondary_color, linewidth=3)
    ax.plot([5.5, 5.7], [8.3, 8.8], color=secondary_color, linewidth=3)
    
    # Signal waves
    wave1 = Circle((4.3, 8.8), 0.1, facecolor=accent_color, edgecolor='none', alpha=0.8)
    wave2 = Circle((5.7, 8.8), 0.1, facecolor=accent_color, edgecolor='none', alpha=0.8)
    ax.add_patch(wave1)
    ax.add_patch(wave2)
    
    # Currency symbols around the bot
    ax.text(2.0, 7.0, '$', fontsize=16, fontweight='bold', 
            color=accent_color, ha='center', va='center', alpha=0.6)
    ax.text(8.0, 7.0, '€', fontsize=16, fontweight='bold', 
            color=secondary_color, ha='center', va='center', alpha=0.6)
    ax.text(2.0, 5.0, '¥', fontsize=16, fontweight='bold', 
            color=red_color, ha='center', va='center', alpha=0.6)
    ax.text(8.0, 5.0, '£', fontsize=16, fontweight='bold', 
            color=primary_color, ha='center', va='center', alpha=0.6)
    
    # Save as high-resolution image
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    buf.seek(0)
    
    # Convert to PIL Image and resize
    img = Image.open(buf)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    plt.close()
    return img

def create_ico_file():
    """Create .ico file with multiple sizes"""
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        img = create_trading_bot_icon(size)
        images.append(img)
    
    # Save as .ico file
    images[0].save('icon.ico', format='ICO', sizes=[(img.width, img.height) for img in images])
    print(f"✅ Created icon.ico with sizes: {sizes}")
    
    # Also save as PNG for other uses
    main_img = create_trading_bot_icon(512)
    main_img.save('trading_bot_logo.png', format='PNG')
    print("✅ Created trading_bot_logo.png (512x512)")
    
    return images

def create_banner():
    """Create a banner/header image for GitHub"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.patch.set_facecolor('#0D1117')  # GitHub dark background
    ax.set_facecolor('#0D1117')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Colors
    primary_color = '#58A6FF'  # GitHub blue
    secondary_color = '#F78166'  # GitHub orange
    accent_color = '#56D364'   # GitHub green
    
    # Title
    ax.text(6, 2.5, 'AI Trading Bot', fontsize=36, fontweight='bold',
            color='white', ha='center', va='center', family='sans-serif')
    
    ax.text(6, 1.8, 'Hybrid ML-DRL Trading System', fontsize=16,
            color=primary_color, ha='center', va='center', family='sans-serif')
    
    # Decorative elements
    # Left side - chart
    x_chart = np.linspace(0.5, 3.5, 50)
    y_chart = 2 + 0.3 * np.sin(x_chart * 3) + 0.05 * x_chart
    ax.plot(x_chart, y_chart, color=accent_color, linewidth=3, alpha=0.8)
    
    # Right side - bot icon
    bot_circle = Circle((10, 2), 0.8, facecolor=primary_color, alpha=0.3, edgecolor=primary_color)
    ax.add_patch(bot_circle)
    ax.text(10, 2, '🤖', fontsize=32, ha='center', va='center')
    
    # Performance indicators
    ax.text(2, 0.8, '🎯 4 Trading Pairs', fontsize=12, color=secondary_color, ha='left')
    ax.text(6, 0.8, '⚡ Real-time Signals', fontsize=12, color=primary_color, ha='left')
    ax.text(9.5, 0.8, '🤖 AI Powered', fontsize=12, color=accent_color, ha='left')
    
    plt.tight_layout()
    plt.savefig('github_banner.png', dpi=300, bbox_inches='tight', 
                facecolor='#0D1117', edgecolor='none')
    plt.close()
    print("✅ Created github_banner.png")

if __name__ == "__main__":
    print("🎨 Creating Trading Bot Icon...")
    
    try:
        # Install required packages if needed
        import matplotlib
        import PIL
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "matplotlib", "Pillow"], check=True)
    
    # Create icons
    create_ico_file()
    create_banner()
    
    print("\n🎉 Icon creation complete!")
    print("📁 Files created:")
    print("   - icon.ico (for executable)")
    print("   - trading_bot_logo.png (high-res logo)")
    print("   - github_banner.png (GitHub header)")
