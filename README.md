# llm_drivenTrader

True AI-Driven Autonomous Trading Bot with Pattern Learning

## 🎯 Overview

A true autonomous trading bot that learns from every trade and adapts to market conditions. Unlike traditional rule-based bots, this system uses pattern recognition and continuously improves its decision-making through real trading experience.

### ✅ Current Performance

- **Win Rate**: 75% (3/4 trades profitable)
- **Net Profit**: $0.051 after fees
- **Learning**: Successfully identifying overbought patterns
- **Risk Management**: Proper stop loss and time limits working

## 🚀 Features

### Pattern-Based Learning

- **Learns from EVERY trade** - Updates pattern memory immediately
- **Market state recognition** - Identifies oversold, overbought, and neutral conditions
- **Dynamic confidence** - Adjusts position sizing based on pattern success rates
- **No dependency on LLMs** - Works reliably with pattern-based decisions

### Smart Entry Logic

```python
# Clear, simple rules that work:
if RSI < 30 and volume > 1.5x:
    action = "LONG"  # Strong oversold signal
elif RSI > 70 and volume > 1.5x:
    action = "SHORT" # Strong overbought signal
else:
    action = "WAIT"  # No clear pattern
```

### Risk Management

- **Stop Loss**: 0.5% (protects capital)
- **Take Profit**: 1.0% (consistent gains)
- **Time Limit**: 5 minutes (prevents overexposure)
- **Position Sizing**: Scales with confidence (50-100% of base size)

### AI Learning System

- Pattern format: `{market_state}_{rsi_bucket}` (e.g., "oversold_2", "overbought_7")
- Tracks success rates and average outcomes
- Improves decisions based on historical performance
- Saves patterns after every trade

## 📋 Requirements

- Python 3.8+
- Bybit API keys (testnet or mainnet)
- Dependencies: `pip install -r requirements.txt`
- LM Studio (optional - bot works without it)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/llm_drivenTrader.git
   cd llm_drivenTrader
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your Bybit API keys
   ```

4. **Initialize patterns**

   ```bash
   python init_patterns.py
   ```

5. **Test setup**
   ```bash
   python test_setup.py
   ```

## 🏃 Running the Bot

### Start Trading

```bash
python main.py
```

### Monitor Performance

```bash
python monitor_AI.py  # Real-time monitoring
python analyze_performance.py  # Performance analysis
```

### Calculate Fees

```bash
python fee_calculator.py  # Shows actual P&L after Bybit fees
```

## 📊 How It Works

### 1. Market Analysis

- Fetches real-time data from Bybit
- Calculates RSI, volume ratio, and other indicators
- Determines market state (oversold/overbought/neutral)

### 2. Pattern Matching

- Checks historical patterns for similar market conditions
- Weighs successful patterns higher in decision-making
- Falls back to technical analysis if no patterns match

### 3. Trade Execution

- Opens positions with calculated stop loss and take profit
- Monitors positions for exit conditions
- Records outcome for pattern learning

### 4. Learning Process

```
Trade Entry → Record Context → Trade Exit → Update Pattern → Save to Memory
     ↓                                           ↓
  (RSI, Volume)                          (Success/Failure)
```

## 💰 Fee Calculation

The bot now accounts for Bybit's trading fees:

- **Maker Fee**: 0.055% per trade
- **Round Trip**: 0.11% total (entry + exit)

To update your bot for fee calculation:

```bash
python patch_fees.py
```

## 📈 Performance Expectations

### Learning Phases

- **First 10 trades**: 40-50% win rate (learning)
- **After 20 trades**: 50-60% win rate (patterns forming)
- **After 50 trades**: 60%+ win rate (optimized)

### Risk/Reward

- Need 34% win rate to break even (with 1:2 risk/reward)
- Current performance: 75% win rate
- Average profit per winning trade: 0.89% (after fees)

## 🔧 Configuration

### Key Settings (.env)

```bash
# Exchange
TESTNET=True  # Use False for mainnet
SYMBOL=SOLUSDT
TIMEFRAME=1

# Position Sizing
DEFAULT_POSITION_SIZE=10
MAX_POSITION_SIZE=100

# Risk Management
MAX_DAILY_LOSS=50
MAX_CONSECUTIVE_LOSSES=3

# LLM (optional)
LLM_URL=http://localhost:1234
LLM_MODEL=local-model
```

## 📁 Project Structure

```
llm_drivenTrader/
├── main.py              # Main bot loop
├── config.py            # Configuration
├── core/
│   ├── bot_engine.py    # Trading engine
│   └── llm_engine.py    # AI decision engine
├── risk/
│   └── risk_management.py
├── strategies/          # Strategy configs (unused in AI mode)
├── ai_patterns.json     # Learned patterns
└── performance_*.json   # Trade history
```

## 🐛 Troubleshooting

### Bot not trading?

- Check RSI levels - bot only trades when RSI < 30 or > 70
- Verify market is open and liquid
- Run `python debug_decision.py` to see decision process

### Wrong P&L shown?

- Run `python patch_fees.py` to include fee calculations
- Check with `python fee_calculator.py`

### Using old version?

- Run `python verify_fix.py` to check
- Look for entry reasons like "Overbought with volume" (old) vs "Overbought RSI:70" (new)

## 🚀 Future Improvements

While keeping the code simple, potential enhancements:

- [ ] Multi-timeframe analysis
- [ ] More sophisticated pattern matching
- [ ] Volatility-based position sizing
- [ ] Additional exchange support

## ⚠️ Disclaimer

This bot is for educational purposes. Trading cryptocurrency involves risk. Always:

- Start with testnet
- Use small position sizes
- Monitor performance closely
- Never invest more than you can afford to lose

## 📊 Live Results

Latest session (2025-07-21):

- 4 trades executed
- 3 wins, 1 loss (75% win rate)
- $0.135 gross profit
- $0.051 net profit after fees
- Successfully identified overbought conditions

## 🤝 Contributing

Contributions welcome! Please:

- Keep code simple and direct
- Fix issues without overcomplicating
- Test thoroughly before submitting PR
- Document any new patterns discovered

## 📄 License

MIT License - See LICENSE file for details

---

**Built with the philosophy**: _Simple code that learns is better than complex code that doesn't._
