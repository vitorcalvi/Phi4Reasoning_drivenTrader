"""
Risk Management Module
"""
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.min_risk_reward_ratio = config.MIN_RISK_REWARD_RATIO
        
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
    
    def initialize(self):
        """Initialize risk manager"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
        logger.info("✅ Risk Manager initialized")
    
    async def validate_decision(self, decision: Dict, market_data: Dict, 
                              position: Optional[Dict], performance: Dict) -> Dict:
        """Validate trading decision"""
        
        # Check if trading allowed
        if not self._is_trading_allowed(performance):
            return {
                'action': 'WAIT',
                'reason': 'Risk limits reached',
                'confidence': 1.0
            }
        
        # Validate entry
        if decision['action'] in ['LONG', 'SHORT']:
            # Check position size
            size = decision.get('position_size', self.config.DEFAULT_POSITION_SIZE)
            decision['position_size'] = min(size, self.max_position_size)
            
            # Adjust for consecutive losses
            consecutive_losses = performance.get('consecutive_losses', 0)
            if consecutive_losses > 0:
                reduction = max(0.5, 1 - (consecutive_losses * 0.2))
                decision['position_size'] *= reduction
            
            # Ensure minimum risk/reward
            price = market_data.get('price', 0)
            if price > 0 and 'stop_loss' in decision and 'take_profit' in decision:
                if decision['action'] == 'LONG':
                    risk = price - decision['stop_loss']
                    reward = decision['take_profit'] - price
                else:
                    risk = decision['stop_loss'] - price
                    reward = price - decision['take_profit']
                
                if risk > 0 and reward / risk < self.min_risk_reward_ratio:
                    if decision['action'] == 'LONG':
                        decision['take_profit'] = price + (risk * self.min_risk_reward_ratio)
                    else:
                        decision['take_profit'] = price - (risk * self.min_risk_reward_ratio)
        
        return decision
    
    def _is_trading_allowed(self, performance: Dict) -> bool:
        """Check if trading allowed"""
        
        # Daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.warning(f"🛑 Daily loss limit: ${self.daily_pnl:.2f}")
            return False
        
        # Consecutive losses
        consecutive_losses = performance.get('consecutive_losses', 0)
        if consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"🛑 Max consecutive losses: {consecutive_losses}")
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if datetime.now().date() > self.session_start.date():
            self.daily_pnl = 0
            self.daily_trades = 0
            self.session_start = datetime.now()
    
    def get_risk_status(self) -> Dict:
        """Get risk status"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_limit_remaining': self.max_daily_loss - abs(self.daily_pnl),
            'trading_allowed': abs(self.daily_pnl) < self.max_daily_loss
        }