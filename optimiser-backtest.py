import pandas as pd
import numpy as np

class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, strategy_type: str = 'scalping'):
        self.data = data.set_index('timestamp')  # Assume timestamp is datetime index
        self.strategy_type = strategy_type
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume'].rolling(10).mean()  # Example placeholder

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd_histogram(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_bb_position(self, prices: pd.Series, window: int = 20, n_std: float = 2) -> pd.Series:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + n_std * std
        lower = sma - n_std * std
        return (prices - lower) / (upper - lower)

    def generate_signals(self, params: dict) -> pd.DataFrame:
        df = self.data.copy()
        df['rsi'] = self.calculate_rsi(df['close'], window=14)
        df['macd_hist'] = self.calculate_macd_histogram(df['close'])
        df['bb_pos'] = self.calculate_bb_position(df['close'])

        # Entry confidence for long (scalping example; adapt for momentum)
        if self.strategy_type == 'scalping':
            long_cond = (
                (df['rsi'] < params['rsi_oversold']) * 0.3 +
                (df['volume_ratio'] > params['volume_multiplier']) * 0.2 +
                (df['macd_hist'] > 0) * 0.15 +  # Placeholder for orderbook_imbalance
                (df['bb_pos'] < 0.2) * 0.15
            ) >= params['min_confidence']
            # Similar for short; add filters like spread and volatility
            df['signal'] = np.where(long_cond, 1, 0)  # 1 for long, -1 for short, 0 for none
        # Adapt for Momentum Breakout similarly using SMA, RSI on multiple timeframes, etc.
        return df

    def backtest(self, params: dict, initial_capital: float = 10000) -> dict:
        df = self.generate_signals(params)
        position = 0
        equity = [initial_capital]
        trades = []

        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            if df['signal'].iloc[i] == 1 and position == 0:  # Enter long
                position = initial_capital * 0.01 / price  # Risk 1% example
                entry_price = price
            elif position > 0:  # Check exits
                profit = (price - entry_price) / entry_price
                if profit >= params['take_profit_pct'] or profit <= -params['stop_loss_pct']:
                    trades.append(profit)
                    initial_capital += initial_capital * profit * 0.01  # Update capital
                    position = 0
            equity.append(initial_capital)

        win_rate = sum(t > 0 for t in trades) / len(trades) if trades else 0
        profit_factor = sum(t for t in trades if t > 0) / abs(sum(t for t in trades if t < 0)) if any(t < 0 for t in trades) else float('inf')
        return {'win_rate': win_rate, 'profit_factor': profit_factor, 'final_equity': initial_capital, 'trades': len(trades)}

    def optimize(self):
        # Parameter ranges from your config
        rsi_oversold = range(25, 41, 5)
        volume_mult = np.arange(1.2, 2.6, 0.1)
        tp_pct = np.arange(0.3, 1.1, 0.1)
        sl_pct = np.arange(0.2, 0.51, 0.05)

        best_params = None
        best_metric = 0
        for ro in rsi_oversold:
            for vm in volume_mult:
                for tp in tp_pct:
                    for sl in sl_pct:
                        params = {'rsi_oversold': ro, 'volume_multiplier': vm, 'take_profit_pct': tp, 'stop_loss_pct': sl, 'min_confidence': 0.7}
                        results = self.backtest(params)
                        metric = results['profit_factor'] * results['win_rate']  # Example objective
                        if metric > best_metric:
                            best_metric = metric
                            best_params = params
        return best_params, self.backtest(best_params)

# Usage:
data = pd.read_csv('data/SOLUSDT.csv')  # Load your data
optimizer = StrategyOptimizer(data, 'scalping')
best_params, performance = optimizer.optimize()
print(best_params, performance)
