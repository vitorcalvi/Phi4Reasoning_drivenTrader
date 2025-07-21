#!/usr/bin/env python3
"""
Real-time monitor for AI Trading Bot
Shows current market conditions and decision logic
"""
import asyncio
import json
from datetime import datetime
from config import Config
from core.bot_engine import BotEngine
from core.llm_engine import LLMEngine

async def monitor():
    """Monitor market and AI decisions"""
    config = Config()
    bot_engine = BotEngine(config)
    llm_engine = LLMEngine(config)
    
    print("🔍 AI Trading Bot Monitor")
    print("=" * 60)
    
    # Test connection
    if not await bot_engine.test_connection():
        print("❌ Failed to connect to exchange")
        return
    
    while True:
        try:
            # Get market data
            market_data = await bot_engine.get_market_data()
            if not market_data:
                print("⚠️  No market data")
                await asyncio.sleep(5)
                continue
            
            # Get indicators
            indicators = market_data.get('indicators', {})
            price = indicators.get('price', 0)
            rsi = indicators.get('rsi', 50)
            volume_ratio = indicators.get('volume_ratio', 1)
            sma = indicators.get('sma_20', price)
            spread = market_data.get('orderbook', {}).get('spread_pct', 0)
            
            # Get position
            position = await bot_engine.get_current_position()
            
            # Clear screen
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            
            # Display header
            print(f"🔍 AI Trading Bot Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            # Market info
            print(f"\n📊 MARKET DATA - {config.SYMBOL}")
            print(f"  Price:        ${price:.4f}")
            print(f"  RSI:          {rsi:.1f} {'🔴' if rsi > 70 else '🟢' if rsi < 30 else '⚪'}")
            print(f"  Volume:       {volume_ratio:.2f}x {'📈' if volume_ratio > 1.2 else '📉'}")
            print(f"  SMA20:        ${sma:.4f} ({'Above' if price > sma else 'Below'})")
            print(f"  Spread:       {spread:.3f}%")
            
            # Position info
            print(f"\n💼 POSITION")
            if position:
                pnl = position.get('unrealized_pnl', 0)
                pnl_pct = position.get('pnl_percent', 0)
                duration = position.get('duration_seconds', 0) / 60
                
                print(f"  Side:         {position['side']}")
                print(f"  Entry:        ${position['entry_price']:.4f}")
                print(f"  PnL:          ${pnl:.2f} ({pnl_pct:.2f}%) {'🟢' if pnl > 0 else '🔴'}")
                print(f"  Duration:     {duration:.1f} min")
            else:
                print(f"  No position")
            
            # AI Analysis
            print(f"\n🤖 AI ANALYSIS")
            
            # Determine likely action
            if position:
                if position.get('pnl_percent', 0) >= 1.5:
                    likely_action = "CLOSE (Take Profit)"
                elif position.get('pnl_percent', 0) <= -0.8:
                    likely_action = "CLOSE (Stop Loss)"
                else:
                    likely_action = "WAIT (Holding)"
            else:
                if rsi < 25 and volume_ratio > 1.5:
                    likely_action = "LONG (Strong Oversold)"
                elif rsi > 75 and volume_ratio > 1.5:
                    likely_action = "SHORT (Strong Overbought)"
                elif rsi < 35:
                    likely_action = "LONG (Oversold)"
                elif rsi > 65:
                    likely_action = "SHORT (Overbought)"
                else:
                    likely_action = "WAIT (No Signal)"
            
            print(f"  Likely Action: {likely_action}")
            print(f"  Market State:  {'Oversold' if rsi < 35 else 'Overbought' if rsi > 65 else 'Neutral'}")
            
            # Stats
            performance = await bot_engine.get_performance_metrics()
            print(f"\n📈 PERFORMANCE")
            print(f"  Total Trades:  {performance.get('total_trades', 0)}")
            print(f"  Win Rate:      {performance.get('winning_trades', 0) / max(performance.get('total_trades', 1), 1) * 100:.0f}%")
            print(f"  Total PnL:     ${performance.get('total_pnl', 0):.2f}")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to exit")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")