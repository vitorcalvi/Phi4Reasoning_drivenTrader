"""
Bot Engine with AI Learning Integration
"""
import asyncio
import json
import logging
from datetime import datetime
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
        self.entry_decision = None  # Store entry decision for learning
        self.llm_engine = None  # Will be set by main
        
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'start_balance': 0,
            'current_balance': 0
        }
    
    def set_llm_engine(self, llm_engine):
        """Set reference to LLM engine for learning feedback"""
        self.llm_engine = llm_engine
    
    async def test_connection(self) -> bool:
        """Test exchange connection"""
        try:
            response = self.exchange.get_server_time()
            if response.get('retCode') == 0:
                logger.info(f"✅ Connected to {'testnet' if self.config.TESTNET else 'mainnet'}")
                balance = await self.get_account_balance()
                self.performance_data['start_balance'] = balance
                self.performance_data['current_balance'] = balance
                return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
        return False
    
    async def get_market_data(self) -> Dict:
        """Get market data with indicators"""
        try:
            # Get price data
            response = self.exchange.get_kline(
                category="linear",
                symbol=self.config.SYMBOL,
                interval=self.config.TIMEFRAME,
                limit=50
            )
            
            if response.get('retCode') != 0:
                return {}
            
            klines = response['result']['list']
            if not klines:
                return {}
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert to float64 explicitly for TA-Lib
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
            
            # Remove any NaN values
            df = df.dropna()
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check if we have enough data
            if len(df) < 20:
                logger.warning(f"Not enough data: {len(df)} candles")
                return {}
            
            # Convert to numpy arrays for TA-Lib
            close = np.array(df['close'].values, dtype=np.float64)
            high = np.array(df['high'].values, dtype=np.float64)
            low = np.array(df['low'].values, dtype=np.float64)
            volume = np.array(df['volume'].values, dtype=np.float64)
            
            price = float(close[-1])
            
            # RSI
            try:
                rsi = talib.RSI(close, timeperiod=14)
                current_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
            except:
                current_rsi = 50
            
            # Moving averages
            try:
                sma_20 = talib.SMA(close, timeperiod=20)
                current_sma = float(sma_20[-1]) if not np.isnan(sma_20[-1]) else price
            except:
                current_sma = price
            
            # Volume ratio
            try:
                if len(volume) >= 20:
                    volume_avg = float(talib.SMA(volume, timeperiod=20)[-1])
                    volume_ratio = float(volume[-1]) / volume_avg if volume_avg > 0 else 1
                else:
                    # Not enough data for SMA, compare to mean
                    volume_avg = float(np.mean(volume))
                    volume_ratio = float(volume[-1]) / volume_avg if volume_avg > 0 else 1
                
                # Ensure reasonable bounds and handle very low volumes
                if volume_ratio < 0.01:
                    volume_ratio = 0.1  # Minimum 0.1x for very low volume
                volume_ratio = max(0.1, min(volume_ratio, 10.0))
            except:
                volume_ratio = 1
            
            # ATR for volatility
            try:
                atr = talib.ATR(high, low, close, timeperiod=14)
                current_atr = float(atr[-1]) if not np.isnan(atr[-1]) else 0
                atr_pct = (current_atr / price * 100) if price > 0 else 0
            except:
                current_atr = 0
                atr_pct = 0
            
            # Get orderbook
            spread_pct = 0.05  # default
            try:
                orderbook_response = self.exchange.get_orderbook(
                    category="linear",
                    symbol=self.config.SYMBOL,
                    limit=5
                )
                
                if orderbook_response.get('retCode') == 0:
                    result = orderbook_response['result']
                    bids = result.get('b', [])
                    asks = result.get('a', [])
                    
                    if bids and asks:
                        bid = float(bids[0][0])
                        ask = float(asks[0][0])
                        spread_pct = ((ask - bid) / ask * 100)
            except:
                pass
            
            return {
                'price': price,
                'indicators': {
                    'price': price,
                    'rsi': current_rsi,
                    'sma_20': current_sma,
                    'volume_ratio': volume_ratio,
                    'atr_pct': atr_pct
                },
                'orderbook': {
                    'spread_pct': spread_pct
                },
                'raw_data': {
                    'close_prices': close[-20:].tolist(),
                    'volumes': volume[-20:].tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Market data error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    async def get_current_position(self) -> Optional[Dict]:
        """Get current position with P&L"""
        if not self.current_position:
            return None
        
        market_data = await self.get_market_data()
        if not market_data:
            return self.current_position
        
        current_price = market_data['price']
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
    
    async def get_performance_metrics(self) -> Dict:
        """Get trading performance"""
        metrics = self.performance_data.copy()
        
        if self.trade_history:
            recent = self.trade_history[-10:]
            metrics['recent_trades'] = len(recent)
            metrics['recent_win_rate'] = len([t for t in recent if t['pnl'] > 0]) / len(recent)
            
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
        
        metrics['current_balance'] = await self.get_account_balance()
        return metrics
    
    async def execute_decision(self, decision: Dict):
        """Execute trading decision"""
        action = decision.get('action')
        
        if action == 'LONG' and not self.current_position:
            await self._open_position('LONG', decision)
        elif action == 'SHORT' and not self.current_position:
            await self._open_position('SHORT', decision)
        elif action == 'CLOSE' and self.current_position:
            await self.close_position(decision.get('reason', 'AI Decision'))
    
    async def _open_position(self, side: str, decision: Dict):
        """Open position and store decision for learning"""
        try:
            market_data = await self.get_market_data()
            if not market_data:
                return
            
            price = market_data['price']
            position_size = decision.get('position_size', self.config.DEFAULT_POSITION_SIZE)
            
            # Get symbol info and calculate quantity
            info = await self._get_symbol_info()
            quantity = self._calculate_quantity(position_size, price, info)
            
            # Place order
            order_side = 'Buy' if side == 'LONG' else 'Sell'
            response = self.exchange.place_order(
                category="linear",
                symbol=self.config.SYMBOL,
                side=order_side,
                orderType="Market",
                qty=quantity,
                timeInForce="IOC"
            )
            
            if response.get('retCode') == 0:
                self.current_position = {
                    'side': side,
                    'entry_price': price,
                    'quantity': float(quantity),
                    'entry_time': datetime.now(),
                    'stop_loss': decision.get('stop_loss', price * (0.995 if side == 'LONG' else 1.005)),
                    'take_profit': decision.get('take_profit', price * (1.01 if side == 'LONG' else 0.99)),
                    'entry_reason': decision.get('reason', 'AI Signal')
                }
                
                # Store entry decision for learning
                self.entry_decision = decision
                
                logger.info(f"📈 Opened {side}: {quantity} @ ${price:.4f}")
                logger.info(f"🎯 AI Confidence: {decision.get('confidence', 0):.0%}")
            else:
                logger.error(f"❌ Order failed: {response.get('retMsg')}")
                
        except Exception as e:
            logger.error(f"❌ Open position error: {e}")
    
    async def close_position(self, reason: str):
        """Close position and feed outcome back to AI"""
        if not self.current_position:
            return
        
        try:
            market_data = await self.get_market_data()
            if not market_data:
                return
            
            price = market_data['price']
            
            # Place closing order
            side = 'Sell' if self.current_position['side'] == 'LONG' else 'Buy'
            response = self.exchange.place_order(
                category="linear",
                symbol=self.config.SYMBOL,
                side=side,
                orderType="Market",
                qty=str(self.current_position['quantity']),
                timeInForce="IOC"
            )
            
            if response.get('retCode') == 0:
                # Calculate P&L
                if self.current_position['side'] == 'LONG':
                    pnl = (price - self.current_position['entry_price']) * self.current_position['quantity']
                else:
                    pnl = (self.current_position['entry_price'] - price) * self.current_position['quantity']
                
                pnl_pct = (pnl / (self.current_position['entry_price'] * self.current_position['quantity'])) * 100
                
                # Record trade
                trade = {
                    'side': self.current_position['side'],
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': price,
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
                
                # Feed outcome back to AI for learning
                if self.llm_engine and self.entry_decision:
                    self.llm_engine.record_trade_outcome(self.entry_decision, price, pnl)
                
                logger.info(f"💰 Closed {self.current_position['side']}: ${pnl:.2f} ({pnl_pct:.1f}%) - {reason}")
                logger.info(f"🧠 AI Learning: Pattern recorded")
                
                self.current_position = None
                self.entry_decision = None
            else:
                logger.error(f"❌ Close failed: {response.get('retMsg')}")
                
        except Exception as e:
            logger.error(f"❌ Close position error: {e}")
    
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
        except:
            pass
        
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
                    'min_order_value': 5.0
                }
        except:
            pass
        
        return {
            'min_qty': 0.001,
            'qty_step': 0.001,
            'tick_size': 0.01,
            'min_order_value': 5.0
        }
    
    def _calculate_quantity(self, usdt_amount: float, price: float, info: Dict) -> str:
        """Calculate quantity"""
        raw_qty = usdt_amount / price
        step = info['qty_step']
        qty = float(int(raw_qty / step) * step)
        qty = max(qty, info['min_qty'])
        
        if qty * price < info['min_order_value']:
            qty = (info['min_order_value'] / price) * 1.01
            qty = float(int(qty / step) * step)
        
        step_str = f"{step:g}"
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        return f"{qty:.{decimals}f}"
    
    async def save_performance_data(self):
        """Save performance data and AI patterns"""
        try:
            filename = f"ai_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            data = {
                'performance': self.performance_data,
                'trades': self.trade_history,
                'final_balance': await self.get_account_balance(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save AI patterns
            if self.llm_engine:
                self.llm_engine.save_patterns()
            
            logger.info(f"💾 Saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Save error: {e}")