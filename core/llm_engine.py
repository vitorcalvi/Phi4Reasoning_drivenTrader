"""
Simplified LLM Engine for AI-Driven Trading Decisions
"""
import json
import logging
import yaml
import requests
import re
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.llm_url = f"{config.LLM_URL}/v1/chat/completions"
        self.llm_model = config.LLM_MODEL
        self.strategies = {}
        self.current_strategy = None
        self.decision_history = []
        
    async def test_connection(self) -> bool:
        """Test LLM connection"""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": "1+1"}],
                    "temperature": 0.0,
                    "max_tokens": 10
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✅ LLM connected: {self.llm_model}")
                return True
        except:
            pass
        
        logger.info("📊 Using AI logic without LLM confirmation")
        return True
    
    async def load_strategies(self) -> Dict:
        """Load trading strategies"""
        strategies_dir = Path("strategies")
        
        for file in strategies_dir.glob("*.yaml"):
            try:
                with open(file, 'r') as f:
                    strategy = yaml.safe_load(f)
                    name = strategy.get('name', file.stem)
                    self.strategies[name] = strategy
                    logger.info(f"📚 Loaded: {name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file}: {e}")
        
        self.current_strategy = '1-Minute Scalping'
        return self.strategies
    
    async def make_decision(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """AI-driven trading decision"""
        indicators = market_data.get('indicators', {})
        price = indicators.get('price', 0)
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1)
        spread_pct = market_data.get('orderbook', {}).get('spread_pct', 0.05)
        
        # AI Logic: Calculate decision based on market conditions
        if position:
            # Position management
            pnl_pct = position.get('pnl_percent', 0)
            
            if pnl_pct >= 1.0:  # 1% profit
                decision = {
                    "action": "CLOSE",
                    "confidence": 0.9,
                    "reason": f"Take profit at {pnl_pct:.1f}%"
                }
            elif pnl_pct <= -0.5:  # 0.5% loss
                decision = {
                    "action": "CLOSE", 
                    "confidence": 0.95,
                    "reason": f"Stop loss at {pnl_pct:.1f}%"
                }
            else:
                decision = {
                    "action": "WAIT",
                    "confidence": 0.6,
                    "reason": f"Holding position, PnL: {pnl_pct:.1f}%"
                }
        else:
            # Entry logic
            if rsi < 30 and volume_ratio > 1.5 and spread_pct < 0.05:
                decision = {
                    "action": "LONG",
                    "confidence": 0.8,
                    "reason": f"Oversold RSI {rsi:.0f}, high volume",
                    "position_size": self.config.DEFAULT_POSITION_SIZE,
                    "stop_loss": price * 0.995,
                    "take_profit": price * 1.01
                }
            elif rsi > 70 and volume_ratio > 1.5 and spread_pct < 0.05:
                decision = {
                    "action": "SHORT",
                    "confidence": 0.8,
                    "reason": f"Overbought RSI {rsi:.0f}, high volume",
                    "position_size": self.config.DEFAULT_POSITION_SIZE,
                    "stop_loss": price * 1.005,
                    "take_profit": price * 0.99
                }
            else:
                decision = {
                    "action": "WAIT",
                    "confidence": 0.5,
                    "reason": f"No clear signal (RSI: {rsi:.0f})"
                }
        
        # Try LLM confirmation if available
        if hasattr(self.config, 'SIMPLE_PROMPTS') and self.config.SIMPLE_PROMPTS:
            confirmed = await self._try_llm_confirmation(decision, indicators)
            if confirmed:
                decision = confirmed
        
        # Log decision
        logger.info(f"🤖 AI Decision: {decision['action']} - {decision['reason']}")
        logger.info(f"🎯 Confidence: {decision['confidence']:.0%}")
        
        # Record history
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'price': price,
            'rsi': rsi
        })
        
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return decision
    
    async def _try_llm_confirmation(self, decision: Dict, indicators: Dict) -> Optional[Dict]:
        """Try to get LLM confirmation"""
        try:
            json_str = json.dumps(decision)
            
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": json_str}],
                    "temperature": 0.0,
                    "max_tokens": 150
                },
                timeout=3
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # Extract JSON from response
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
                
                # Find JSON
                match = re.search(r'\{[^{}]*"action"[^{}]*\}', content, re.DOTALL)
                if match:
                    try:
                        confirmed = json.loads(match.group())
                        if self._validate_decision(confirmed):
                            logger.info("✅ LLM confirmed")
                            return confirmed
                    except:
                        pass
        except:
            pass
        
        return None
    
    def _validate_decision(self, decision: Dict) -> bool:
        """Validate decision format"""
        if 'action' not in decision:
            return False
        
        decision['action'] = decision['action'].upper()
        
        if decision['action'] not in ['LONG', 'SHORT', 'CLOSE', 'WAIT']:
            return False
        
        if 'confidence' not in decision:
            decision['confidence'] = 0.5
        
        if 'reason' not in decision:
            decision['reason'] = 'AI decision'
        
        try:
            decision['confidence'] = float(decision['confidence'])
            if not (0 <= decision['confidence'] <= 1):
                return False
        except:
            return False
        
        return True
    
    def get_decision_stats(self) -> Dict:
        """Get decision statistics"""
        if not self.decision_history:
            return {}
        
        recent = self.decision_history[-20:]
        action_counts = {}
        
        for record in recent:
            action = record['decision']['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_decisions': len(recent),
            'action_distribution': action_counts,
            'last_decision': recent[-1] if recent else None
        }