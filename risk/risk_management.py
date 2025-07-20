"""
Risk Management Module
Validates and adjusts trading decisions based on risk parameters
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config):
        self.config = config
        
        # Risk parameters
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.max_drawdown = config.MAX_DRAWDOWN
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.min_risk_reward_ratio = config.MIN_RISK_REWARD_RATIO
        
        # Risk tracking
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
        self.peak_balance = 0
        self.risk_events = []
    
    def initialize(self):
        """Initialize risk parameters"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
        logger.info("✅ Risk Manager initialized")
        logger.info(f"📊 Max position: ${self.max_position_size}")
        logger.info(f"📊 Max daily loss: ${self.max_daily_loss}")
        logger.info(f"📊 Max drawdown: {self.max_drawdown}%")
    
    async def validate_decision(self, decision: Dict, market_data: Dict, 
                              position: Optional[Dict], performance: Dict) -> Dict:
        """Validate and potentially modify trading decision"""
        
        # Create a copy to avoid modifying original
        validated_decision = decision.copy()
        
        # Check if trading is allowed
        if not self._is_trading_allowed(performance):
            return {
                'action': 'WAIT',
                'reason': 'Trading halted due to risk limits',
                'original_decision': decision,
                'risk_override': True
            }
        
        # Validate based on action type
        if decision['action'] in ['LONG', 'SHORT']:
            validated_decision = self._validate_entry(
                validated_decision, market_data, performance
            )
        elif decision['action'] == 'CLOSE' and position:
            validated_decision = self._validate_exit(
                validated_decision, position, market_data
            )
        
        # Log risk decision
        if validated_decision != decision:
            logger.warning(f"⚠️ Risk override: {decision['action']} → {validated_decision['action']}")
            self._log_risk_event('decision_override', validated_decision)
        
        return validated_decision
    
    def _is_trading_allowed(self, performance: Dict) -> bool:
        """Check if trading should be allowed"""
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.warning(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
            self._log_risk_event('daily_loss_limit', {'daily_pnl': self.daily_pnl})
            return False
        
        # Check consecutive losses
        consecutive_losses = performance.get('consecutive_losses', 0)
        if consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"🛑 Max consecutive losses reached: {consecutive_losses}")
            self._log_risk_event('consecutive_losses', {'count': consecutive_losses})
            return False
        
        # Check drawdown
        if self.peak_balance > 0:
            current_balance = performance.get('current_balance', 0)
            drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            
            if drawdown_pct >= self.max_drawdown:
                logger.warning(f"🛑 Max drawdown reached: {drawdown_pct:.1f}%")
                self._log_risk_event('max_drawdown', {'drawdown_pct': drawdown_pct})
                return False
        
        # Check time-based restrictions
        current_hour = datetime.now().hour
        if current_hour in [0, 1, 2, 3]:  # Avoid low liquidity hours
            logger.info("⏰ Trading restricted during low liquidity hours")
            return False
        
        return True
    
    def _validate_entry(self, decision: Dict, market_data: Dict, performance: Dict) -> Dict:
        """Validate entry decision"""
        
        # 1. Position sizing
        original_size = decision.get('position_size', self.config.DEFAULT_POSITION_SIZE)
        validated_size = self._calculate_position_size(
            original_size, 
            decision.get('confidence', 0.5),
            performance
        )
        
        decision['position_size'] = validated_size
        
        # 2. Risk/Reward validation
        price = market_data.get('price', 0)
        stop_loss = decision.get('stop_loss', 0)
        take_profit = decision.get('take_profit', 0)
        
        if price > 0 and stop_loss > 0 and take_profit > 0:
            if decision['action'] == 'LONG':
                risk = price - stop_loss
                reward = take_profit - price
            else:  # SHORT
                risk = stop_loss - price
                reward = price - take_profit
            
            if risk > 0:
                risk_reward_ratio = reward / risk
                
                if risk_reward_ratio < self.min_risk_reward_ratio:
                    # Adjust take profit to meet minimum ratio
                    if decision['action'] == 'LONG':
                        decision['take_profit'] = price + (risk * self.min_risk_reward_ratio)
                    else:
                        decision['take_profit'] = price - (risk * self.min_risk_reward_ratio)
                    
                    decision['reason'] += f" (TP adjusted for {self.min_risk_reward_ratio}:1 RR)"
        
        # 3. Spread check
        spread_pct = market_data.get('orderbook', {}).get('spread_pct', 0)
        if spread_pct > 0.1:  # 0.1% spread
            if decision.get('confidence', 0.5) < 0.8:
                decision['action'] = 'WAIT'
                decision['reason'] = f"Spread too high ({spread_pct:.3f}%)"
                decision['risk_override'] = True
        
        # 4. Volatility check
        atr_pct = market_data.get('indicators', {}).get('atr_pct', 0)
        if atr_pct > 5:  # 5% ATR
            # Reduce position size in high volatility
            decision['position_size'] *= 0.5
            decision['reason'] += " (Size reduced due to high volatility)"
        
        # 5. Reduce size after losses
        consecutive_losses = performance.get('consecutive_losses', 0)
        if consecutive_losses > 0:
            reduction_factor = max(0.3, 1 - (consecutive_losses * 0.2))
            decision['position_size'] *= reduction_factor
            decision['reason'] += f" (Size reduced after {consecutive_losses} losses)"
        
        return decision
    
    def _validate_exit(self, decision: Dict, position: Dict, market_data: Dict) -> Dict:
        """Validate exit decision"""
        
        # Allow profitable exits
        if position.get('pnl_percent', 0) > 0:
            return decision
        
        # Check if loss is acceptable
        potential_loss = abs(position.get('unrealized_pnl', 0))
        
        # Force exit if loss is too large
        if potential_loss > self.max_position_size * 0.02:  # 2% of max position
            decision['reason'] = "Emergency exit - loss limit"
            decision['risk_override'] = True
        
        return decision
    
    def _calculate_position_size(self, requested_size: float, confidence: float, 
                               performance: Dict) -> float:
        """Calculate appropriate position size based on risk factors"""
        
        # Start with requested size
        size = min(requested_size, self.max_position_size)
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1x based on confidence
        size *= confidence_multiplier
        
        # Kelly Criterion inspired sizing
        win_rate = performance.get('recent_win_rate', 0.5)
        if win_rate > 0 and win_rate < 1:
            avg_win = abs(performance.get('recent_avg_pnl', 0))
            if avg_win > 0:
                # Simplified Kelly: f = p - q/b
                # where p = win rate, q = loss rate, b = avg win/loss ratio
                kelly_fraction = win_rate - (1 - win_rate)
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                
                size *= (1 + kelly_fraction)
        
        # Ensure minimum size
        size = max(size, self.config.MIN_POSITION_SIZE)
        
        # Round to reasonable precision
        size = round(size, 2)
        
        return size
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # Check if new day
        if datetime.now().date() > self.session_start.date():
            self._reset_daily_stats()
    
    def update_peak_balance(self, balance: float):
        """Update peak balance for drawdown calculation"""
        if balance > self.peak_balance:
            self.peak_balance = balance
    
    def _reset_daily_stats(self):
        """Reset daily statistics"""
        logger.info(f"📊 Daily summary: {self.daily_trades} trades, P&L: ${self.daily_pnl:.2f}")
        
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
    
    def _log_risk_event(self, event_type: str, details: Dict):
        """Log risk events for analysis"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        }
        
        self.risk_events.append(event)
        
        # Keep only recent events
        if len(self.risk_events) > 100:
            self.risk_events = self.risk_events[-100:]
    
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_limit_remaining': self.max_daily_loss - abs(self.daily_pnl),
            'current_drawdown': self._calculate_current_drawdown(),
            'risk_events_today': len([e for e in self.risk_events 
                                    if e['timestamp'].startswith(datetime.now().date().isoformat())]),
            'trading_allowed': self._is_trading_allowed({'current_balance': self.peak_balance})
        }
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_balance <= 0:
            return 0
        
        # This would need actual balance from bot engine
        # For now, return 0
        return 0
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        self._log_risk_event('emergency_stop', {'reason': 'Manual intervention'})
        
        # Set flags to prevent any trading
        self.max_daily_loss = 0
        self.max_consecutive_losses = 0