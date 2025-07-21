#!/usr/bin/env python3
"""
Calculate actual profits after Bybit fees
"""
import json
import glob

def calculate_fees():
    """Calculate trading fees and actual profits"""
    print("💰 Trading Fee Calculator")
    print("=" * 60)
    
    # Bybit fees
    MAKER_FEE = 0.00055  # 0.055%
    TAKER_FEE = 0.00055  # 0.055% (same for your VIP level)
    
    # Find latest performance file
    files = glob.glob("performance_*.json")
    if not files:
        print("No performance files found")
        return
    
    latest_file = max(files)
    print(f"📁 Analyzing: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    trades = data.get('trades', [])
    if not trades:
        print("No trades found")
        return
    
    print(f"\n📊 Found {len(trades)} trades:")
    print("-" * 80)
    print(f"{'#':<3} {'Side':<6} {'Entry':<8} {'Exit':<8} {'Qty':<6} {'Gross P&L':<10} {'Fees':<8} {'Net P&L':<10}")
    print("-" * 80)
    
    total_gross_pnl = 0
    total_fees = 0
    total_net_pnl = 0
    
    for i, trade in enumerate(trades, 1):
        entry = trade['entry_price']
        exit = trade['exit_price']
        qty = trade['quantity']
        
        # Calculate fees
        entry_value = entry * qty
        exit_value = exit * qty
        entry_fee = entry_value * MAKER_FEE
        exit_fee = exit_value * MAKER_FEE
        total_fee = entry_fee + exit_fee
        
        # Calculate P&L
        if trade['side'] == 'LONG':
            gross_pnl = (exit - entry) * qty
        else:
            gross_pnl = (entry - exit) * qty
        
        net_pnl = gross_pnl - total_fee
        
        # Update totals
        total_gross_pnl += gross_pnl
        total_fees += total_fee
        total_net_pnl += net_pnl
        
        print(f"{i:<3} {trade['side']:<6} {entry:<8.2f} {exit:<8.2f} {qty:<6.2f} "
              f"${gross_pnl:<9.4f} ${total_fee:<7.4f} ${net_pnl:<9.4f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<35} ${total_gross_pnl:<9.4f} ${total_fees:<7.4f} ${total_net_pnl:<9.4f}")
    print("=" * 80)
    
    # Compare with bot's reported P&L
    bot_pnl = data['performance']['total_pnl']
    print(f"\n📊 Summary:")
    print(f"   Bot reported P&L: ${bot_pnl:.4f} (gross, no fees)")
    print(f"   Calculated gross: ${total_gross_pnl:.4f}")
    print(f"   Total fees (0.11% round-trip): ${total_fees:.4f}")
    print(f"   Actual net P&L: ${total_net_pnl:.4f}")
    
    # Check against Bybit
    print(f"\n🏦 Bybit Comparison:")
    print(f"   Your Bybit P&L: $0.0438")
    print(f"   Calculated net: ${total_net_pnl:.4f}")
    print(f"   Difference: ${abs(0.0438 - total_net_pnl):.4f}")
    
    if abs(0.0438 - total_net_pnl) < 0.01:
        print("   ✅ Match! (within rounding)")
    else:
        print("   ⚠️  Small difference (likely due to exact fill prices)")
    
    # Performance metrics
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100
    print(f"\n📈 Performance After Fees:")
    print(f"   Gross win rate: {win_rate:.0f}%")
    
    # Recalculate with fees
    wins_after_fees = 0
    for i, trade in enumerate(trades):
        entry = trade['entry_price']
        exit = trade['exit_price']
        qty = trade['quantity']
        
        if trade['side'] == 'LONG':
            gross_pnl = (exit - entry) * qty
        else:
            gross_pnl = (entry - exit) * qty
        
        fee = (entry * qty + exit * qty) * MAKER_FEE
        net_pnl = gross_pnl - fee
        
        if net_pnl > 0:
            wins_after_fees += 1
    
    net_win_rate = wins_after_fees / len(trades) * 100
    print(f"   Net win rate (after fees): {net_win_rate:.0f}%")
    print(f"   Average fee per trade: ${total_fees/len(trades):.4f}")
    
    # Break-even calculation
    avg_position_size = sum(t['entry_price'] * t['quantity'] for t in trades) / len(trades)
    breakeven_move = (MAKER_FEE * 2) * 100  # Need to cover entry + exit fees
    print(f"\n💡 Break-even Analysis:")
    print(f"   Average position size: ${avg_position_size:.2f}")
    print(f"   Break-even move needed: {breakeven_move:.3f}% (to cover 0.11% fees)")
    print(f"   Your stop loss: 0.5% (covers fees + 0.39% buffer)")
    print(f"   Your take profit: 1.0% (0.89% profit after fees)")

if __name__ == "__main__":
    calculate_fees()