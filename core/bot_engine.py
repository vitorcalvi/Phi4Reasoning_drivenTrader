"""
Bot Engine - Handles market data, exchange interface, and trade execution
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import talib
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


class BotEngine:
    def __init__(self, config):
        self.config = config
        self.exchange = HTTP(
            demo=config.TESTNET,
            api_key=config.API_KEY,
            api_secret=config.SECRET_KEY
        )
        
        self.current_position = None
        self.trade_history = []
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'start_balance': 0,
            'current_balance': 0
        }
    
    async def test_connection(self) -> bool:
        """Test exchange connection"""
        try:
            response = self.exchange.get_server_time()
            if response.get('retCode') == 0:
                logger.info(f"✅ Connected to {'testnet' if self.config.TESTNET else 'mainnet'}")
                
                # Get initial balance
                balance = await self.get_account_balance()
                self.performance_data['start_balance'] = balance
                self.performance_data['current_balance'] = balance
                
                return True
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
        return False
    
    async def get_market_data(self) -> Dict:
        """Get comprehensive market data for LLM analysis"""
        try:
            data = {}
            
            # 1. Price data and indicators
            klines = await self._get_klines()
            if klines.empty:
                return {}
            
            data.update(self._calculate_indicators(klines))
            
            # 2. Order book
            orderbook_data = await self._get_orderbook_data()
            data.update(orderbook_data)
            
            # 3. Recent trades
            trades_data = await self._get_recent_trades()
            data.update(trades_data)
            
            # 4. Funding rate
            funding_data = await self._get_funding_rate()
            data.update(funding_data)
            
            # 5. Market sentiment
            data['timestamp'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return {}
    
    async def _get_klines(self) -> pd.DataFrame:
        """Get kline data"""
        try:
            response = self.exchange.get_kline(
                category="linear",
                symbol=self.config.SYMBOL,
                interval=self.config.TIMEFRAME,
                limit=200
            )
            
            if response.get('retCode') == 0:
                klines = response['result']['list']
                if not klines:
                    return pd.DataFrame()
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
                
        except Exception as e:
            logger.error(f"❌ Error getting klines: {e}")
        
        return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if len(df) < 20:
            return {'price': 0, 'indicators': {}}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # Price info
        indicators['price'] = float(close[-1])
        indicators['price_change_1'] = float(close[-1] - close[-2]) if len(close) > 1 else 0
        indicators['price_change_pct_1'] = (indicators['price_change_1'] / close[-2] * 100) if len(close) > 1 and close[-2] > 0 else 0
        
        # Moving averages
        indicators['sma_20'] = float(talib.SMA(close, timeperiod=20)[-1])
        indicators['ema_20'] = float(talib.EMA(close, timeperiod=20)[-1])
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        indicators['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
        indicators['rsi_prev'] = float(rsi[-2]) if len(rsi) > 1 and not np.isnan(rsi[-2]) else 50
        
        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0
        indicators['macd_signal'] = float(signal[-1]) if not np.isnan(signal[-1]) else 0
        indicators['macd_histogram'] = float(hist[-1]) if not np.isnan(hist[-1]) else 0
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        indicators['bb_upper'] = float(upper[-1]) if not np.isnan(upper[-1]) else indicators['price']
        indicators['bb_middle'] = float(middle[-1]) if not np.isnan(middle[-1]) else indicators['price']
        indicators['bb_lower'] = float(lower[-1]) if not np.isnan(lower[-1]) else indicators['price']
        indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        indicators['bb_position'] = (indicators['price'] - indicators['bb_lower']) / indicators['bb_width'] if indicators['bb_width'] > 0 else 0.5
        
        # Volume
        indicators['volume'] = float(volume[-1])
        indicators['volume_sma'] = float(talib.SMA(volume, timeperiod=20)[-1]) if len(volume) >= 20 else float(volume[-1])
        indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=14)
        indicators['atr'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0
        indicators['atr_pct'] = (indicators['atr'] / indicators['price'] * 100) if indicators['price'] > 0 else 0
        
        # Support/Resistance
        indicators['resistance'] = float(high[-20:].max())
        indicators['support'] = float(low[-20:].min())
        
        return {
            'price': indicators['price'],
            'indicators': indicators,
            'candles': {
                'last_5': {
                    'close': close[-5:].tolist(),
                    'volume': volume[-5:].tolist()
                }
            }
        }
    
    async def _get_orderbook_data(self) -> Dict:
        """Get order book data"""
        try:
            response = self.exchange.get_orderbook(
                category="linear",
                symbol=self.config.SYMBOL,
                limit=25
            )
            
            if response.get('retCode') == 0:
                result = response['result']
                bids = result.get('b', [])
                asks = result.get('a', [])
                
                if bids and asks:
                    bid_volume = sum(float(bid[1]) for bid in bids[:5])
                    ask_volume = sum(float(ask[1]) for ask in asks[:5])
                    
                    return {
                        'orderbook': {
                            'bid_price': float(bids[0][0]),
                            'ask_price': float(asks[0][0]),
                            'spread': float(asks[0][0]) - float(bids[0][0]),
                            'spread_pct': ((float(asks[0][0]) - float(bids[0][0])) / float(asks[0][0]) * 100),
                            'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
                            'bid_volume_5': bid_volume,
                            'ask_volume_5': ask_volume
                        }
                    }
        except Exception as e:
            logger.error(f"❌ Error getting orderbook: {e}")
        
        return {'orderbook': {}}
    
    async def _get_recent_trades(self) -> Dict:
        """Get recent trades data"""
        try:
            response = self.exchange.get_public_trade_history(
                category="linear",
                symbol=self.config.SYMBOL,
                limit=100
            )
            
            if response.get('retCode') == 0:
                trades = response['result'].get('list', [])
                
                if trades:
                    buy_volume = sum(float(t['size']) for t in trades if t['side'] == 'Buy')
                    sell_volume = sum(float(t['size']) for t in trades if t['side'] == 'Sell')
                    
                    trade_sizes = [float(t['size']) for t in trades]
                    avg_size = sum(trade_sizes) / len(trade_sizes)
                    large_trades = len([s for s in trade_sizes if s > avg_size * 3])
                    
                    return {
                        'trades': {
                            'buy_volume': buy_volume,
                            'sell_volume': sell_volume,
                            'buy_sell_ratio': buy_volume / (sell_volume + 0.0001),
                            'large_trades_count': large_trades,
                            'avg_trade_size': avg_size
                        }
                    }
        except Exception as e:
            logger.error(f"❌ Error getting trades: {e}")
        
        return {'trades': {}}
    
    async def _get_funding_rate(self) -> Dict:
        """Get funding rate"""
        try:
            response = self.exchange.get_funding_rate_history(
                category="linear",
                symbol=self.config.SYMBOL,
                limit=1
            )
            
            if response.get('retCode') == 0:
                funding_list = response['result'].get('list', [])
                if funding_list:
                    rate = float(funding_list[0]['fundingRate'])
                    return {
                        'funding': {
                            'rate': rate,
                            'rate_pct': rate * 100,
                            'bias': 'long_heavy' if rate > 0.001 else 'short_heavy' if rate < -0.001 else 'neutral'
                        }
                    }
        except Exception as e:
            logger.error(f"❌ Error getting funding rate: {e}")
        
        return {'funding': {}}
    
    async def get_current_position(self) -> Optional[Dict]:
        """Get current position with live P&L"""
        if not self.current_position:
            return None
        
        # Update with current market price
        market_data = await self._get_klines()
        if not market_data.empty:
            current_price = float(market_data['close'].iloc[-1])
            
            position = self.current_position.copy()
            
            # Calculate P&L
            if position['side'] == 'LONG':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                position['pnl_percent'] = ((current_price - position['entry_price']) / position['entry_price'] * 100)
            else:
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                position['pnl_percent'] = ((position['entry_price'] - current_price) / position['entry_price'] * 100)
            
            position['current_price'] = current_price
            position['duration_seconds'] = (datetime.now() - position['entry_time']).total_seconds()
            
            return position
        
        return self.current_position
    
    async def get_performance_metrics(self) -> Dict:
        """Get trading performance metrics"""
        metrics = self.performance_data.copy()
        
        # Add recent trades info
        if self.trade_history:
            recent_trades = self.trade_history[-10:]
            metrics['recent_trades'] = len(recent_trades)
            metrics['recent_win_rate'] = len([t for t in recent_trades if t['pnl'] > 0]) / len(recent_trades)
            metrics['recent_avg_pnl'] = sum(t['pnl'] for t in recent_trades) / len(recent_trades)
            
            # Consecutive wins/losses
            if self.trade_history:
                last_trade = self.trade_history[-1]
                consecutive = 1
                for i in range(len(self.trade_history) - 2, -1, -1):
                    if (self.trade_history[i]['pnl'] > 0) == (last_trade['pnl'] > 0):
                        consecutive += 1
                    else:
                        break
                
                if last_trade['pnl'] > 0:
                    metrics['consecutive_wins'] = consecutive
                    metrics['consecutive_losses'] = 0
                else:
                    metrics['consecutive_wins'] = 0
                    metrics['consecutive_losses'] = consecutive
        
        # Update current balance
        metrics['current_balance'] = await self.get_account_balance()
        
        return metrics
    
    async def execute_decision(self, decision: Dict):
        """Execute trading decision from LLM"""
        action = decision.get('action')
        
        if action == 'LONG' and not self.current_position:
            await self._open_position('LONG', decision)
        elif action == 'SHORT' and not self.current_position:
            await self._open_position('SHORT', decision)
        elif action == 'CLOSE' and self.current_position:
            await self.close_position(decision.get('reason', 'LLM Decision'))
        elif action == 'UPDATE_SL' and self.current_position:
            # Update stop loss if needed
            self.current_position['stop_loss'] = decision.get('stop_loss', self.current_position['stop_loss'])
        elif action == 'UPDATE_TP' and self.current_position:
            # Update take profit if needed
            self.current_position['take_profit'] = decision.get('take_profit', self.current_position['take_profit'])
    
    async def _open_position(self, side: str, decision: Dict):
        """Open a new position"""
        try:
            # Get current price
            market_data = await self._get_klines()
            if market_data.empty:
                return
            
            current_price = float(market_data['close'].iloc[-1])
            
            # Get position size from decision
            position_size_usdt = decision.get('position_size', self.config.DEFAULT_POSITION_SIZE)
            
            # Get symbol info
            info = await self._get_symbol_info()
            quantity = self._calculate_quantity(position_size_usdt, current_price, info)
            
            # Place order
            order_side = 'Buy' if side == 'LONG' else 'Sell'
            order = await self._place_order(order_side, quantity)
            
            if order:
                self.current_position = {
                    'side': side,
                    'entry_price': current_price,
                    'quantity': float(quantity),
                    'entry_time': datetime.now(),
                    'stop_loss': decision.get('stop_loss', current_price * (0.995 if side == 'LONG' else 1.005)),
                    'take_profit': decision.get('take_profit', current_price * (1.01 if side == 'LONG' else 0.99)),
                    'order_id': order.get('orderId'),
                    'entry_reason': decision.get('reason', 'LLM Signal')
                }
                
                logger.info(f"📈 Opened {side}: {quantity} @ ${current_price:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error opening position: {e}")
    
    async def close_position(self, reason: str):
        """Close current position"""
        if not self.current_position:
            return
        
        try:
            # Get current price
            market_data = await self._get_klines()
            if market_data.empty:
                return
            
            current_price = float(market_data['close'].iloc[-1])
            
            # Place closing order
            side = 'Sell' if self.current_position['side'] == 'LONG' else 'Buy'
            order = await self._place_order(side, str(self.current_position['quantity']))
            
            if order:
                # Calculate P&L
                if self.current_position['side'] == 'LONG':
                    pnl = (current_price - self.current_position['entry_price']) * self.current_position['quantity']
                else:
                    pnl = (self.current_position['entry_price'] - current_price) * self.current_position['quantity']
                
                pnl_pct = (pnl / (self.current_position['entry_price'] * self.current_position['quantity'])) * 100
                
                # Record trade
                trade = {
                    'side': self.current_position['side'],
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': current_price,
                    'quantity': self.current_position['quantity'],
                    'pnl': pnl,
                    'pnl_percent': pnl_pct,
                    'entry_time': self.current_position['entry_time'].isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.current_position['entry_time']).total_seconds() / 60,
                    'entry_reason': self.current_position.get('entry_reason', 'Unknown'),
                    'exit_reason': reason
                }
                
                self.trade_history.append(trade)
                
                # Update performance
                self.performance_data['total_trades'] += 1
                if pnl > 0:
                    self.performance_data['winning_trades'] += 1
                self.performance_data['total_pnl'] += pnl
                
                logger.info(f"🔒 Closed {self.current_position['side']}: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {reason}")
                
                self.current_position = None
                
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
    
    async def check_exit_conditions(self):
        """Check if position should be closed based on SL/TP"""
        if not self.current_position:
            return
        
        position = await self.get_current_position()
        if not position:
            return
        
        current_price = position['current_price']
        
        # Check stop loss
        if position['side'] == 'LONG' and current_price <= position['stop_loss']:
            await self.close_position('Stop Loss')
        elif position['side'] == 'SHORT' and current_price >= position['stop_loss']:
            await self.close_position('Stop Loss')
        
        # Check take profit
        elif position['side'] == 'LONG' and current_price >= position['take_profit']:
            await self.close_position('Take Profit')
        elif position['side'] == 'SHORT' and current_price <= position['take_profit']:
            await self.close_position('Take Profit')
    
    async def get_account_balance(self) -> float:
        """Get USDT balance"""
        try:
            response = self.exchange.get_wallet_balance(accountType="UNIFIED")
            
            if response.get('retCode') == 0:
                accounts = response.get('result', {}).get('list', [])
                for account in accounts:
                    coins = account.get('coin', [])
                    for coin in coins:
                        if coin.get('coin') == 'USDT':
                            return float(coin.get('availableBalance', 0))
        except Exception as e:
            logger.error(f"⚠️ Balance check error: {e}")
        
        return 0
    
    async def _get_symbol_info(self) -> Dict:
        """Get symbol trading rules"""
        try:
            response = self.exchange.get_instruments_info(
                category="linear",
                symbol=self.config.SYMBOL
            )
            
            if response.get('retCode') == 0 and response['result']['list']:
                info = response['result']['list'][0]
                return {
                    'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                    'qty_step': float(info['lotSizeFilter']['qtyStep']),
                    'tick_size': float(info['priceFilter']['tickSize']),
                    'min_order_value': 5.0  # Bybit minimum
                }
        except Exception as e:
            logger.error(f"⚠️ Symbol info error: {e}")
        
        return {
            'min_qty': 0.001,
            'qty_step': 0.001,
            'tick_size': 0.01,
            'min_order_value': 5.0
        }
    
    def _calculate_quantity(self, usdt_amount: float, price: float, info: Dict) -> str:
        """Calculate properly formatted quantity"""
        raw_qty = usdt_amount / price
        
        # Round to step
        step = info['qty_step']
        qty = float(int(raw_qty / step) * step)
        
        # Ensure minimum
        qty = max(qty, info['min_qty'])
        
        # Ensure minimum order value
        if qty * price < info['min_order_value']:
            qty = (info['min_order_value'] / price) * 1.01
            qty = float(int(qty / step) * step)
        
        # Format
        step_str = f"{step:g}"
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        return f"{qty:.{decimals}f}"
    
    async def _place_order(self, side: str, quantity: str) -> Optional[Dict]:
        """Place market order"""
        try:
            response = self.exchange.place_order(
                category="linear",
                symbol=self.config.SYMBOL,
                side=side,
                orderType="Market",
                qty=quantity,
                timeInForce="IOC"
            )
            
            if response.get('retCode') == 0:
                return response.get('result', {})
            else:
                logger.error(f"❌ Order failed: {response.get('retMsg', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Order error: {e}")
        
        return None
    
    async def save_performance_data(self):
        """Save performance data to file"""
        try:
            filename = f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            data = {
                'performance': self.performance_data,
                'trades': self.trade_history,
                'final_balance': await self.get_account_balance(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"💾 Performance data saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving performance data: {e}")