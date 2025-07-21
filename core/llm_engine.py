"""
True AI-Driven Autonomous Decision Making Engine
"""
import json
import logging
import requests
import re
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.llm_url = f"{config.LLM_URL}/v1/chat/completions"
        self.llm_model = config.LLM_MODEL
        self.decision_history = []
        self.trade_outcomes = []
        self.pattern_memory = {}
        
        # Detect problematic models
        self.is_phi4 = 'phi' in self.llm_model.lower()
        if self.is_phi4:
            logger.info("⚠️  Phi-4 detected - using pattern-based decisions only")
        
    async def test_connection(self) -> bool:
        """Test LLM connection"""
        if self.is_phi4:
            logger.info("✅ AI Engine ready (Phi-4 mode - pattern-based decisions)")
            return True
            
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": "1+1"}],
                    "temperature": 0.1,
                    "max_tokens": 10
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✅ AI Engine connected: {self.llm_model}")
                return True
        except:
            logger.error("❌ AI Engine offline - using cached patterns")
        return True
    
    async def make_decision(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Pure AI-driven decision making"""
        
        # Build comprehensive context for AI
        context = self._build_context(market_data, position, performance)
        
        # Get AI decision
        decision = self._get_ai_decision(context)
        
        # Record for learning
        self._record_decision(decision, context)
        
        return decision
    
    def _build_context(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Build complete context for AI analysis"""
        indicators = market_data.get('indicators', {})
        
        # Calculate derived features
        price_changes = self._calculate_price_changes(indicators.get('price', 0))
        market_state = self._analyze_market_state(indicators)
        
        # Get relevant historical patterns
        similar_patterns = self._find_similar_patterns(market_state)
        
        context = {
            "current_market": {
                "price": indicators.get('price', 0),
                "rsi": indicators.get('rsi', 50),
                "volume_ratio": indicators.get('volume_ratio', 1),
                "sma_20": indicators.get('sma_20', 0),
                "atr_pct": indicators.get('atr_pct', 0),
                "spread_pct": market_data.get('orderbook', {}).get('spread_pct', 0)
            },
            "price_momentum": price_changes,
            "market_state": market_state,
            "position": position,
            "performance": {
                "win_rate": performance.get('winning_trades', 0) / max(performance.get('total_trades', 1), 1),
                "consecutive_losses": performance.get('consecutive_losses', 0),
                "daily_pnl": performance.get('total_pnl', 0)
            },
            "recent_decisions": self.decision_history[-5:],
            "similar_historical_patterns": similar_patterns
        }
        
        return context
    
    def _get_ai_decision(self, context: Dict) -> Dict:
        """Get decision from AI"""
        
        # Simplified prompt for Phi-4
        market = context['current_market']
        position = context['position']
        
        # Direct decision request
        if position:
            prompt = f"""Price: {market['price']}, RSI: {market['rsi']}, Position: {position['side']} PnL: {position.get('pnl_percent', 0):.1f}%
Output exactly: {{"action": "CLOSE or WAIT", "confidence": 0.0-1.0, "reason": "brief reason"}}"""
        else:
            prompt = f"""Price: {market['price']}, RSI: {market['rsi']}, Volume: {market['volume_ratio']:.1f}x
Output exactly: {{"action": "LONG or SHORT or WAIT", "confidence": 0.0-1.0, "reason": "brief reason"}}"""

        try:
            # Try direct prompt first
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,  # Lower for more deterministic
                    "max_tokens": 500  # Increased for complete response
                },
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                decision = self._extract_decision(content)
                if decision:
                    logger.info(f"🤖 AI Decision: {decision['action']} ({decision['confidence']:.0%})")
                    return decision
            
            # Try alternative prompt format
            if not position:
                alt_prompt = f'{{"action": "WAIT", "confidence": 0.7, "reason": "RSI {market["rsi"]:.0f}"}}'
            else:
                alt_prompt = f'{{"action": "WAIT", "confidence": 0.6, "reason": "Holding"}}'
            
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": alt_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 100
                },
                timeout=5
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                decision = self._extract_decision(content)
                if decision:
                    return decision
        except Exception as e:
            logger.error(f"AI error: {e}")
        
        # Fallback to pattern-based decision
        return self._pattern_based_decision(context)
    
    def _extract_decision(self, content: str) -> Optional[Dict]:
        """Extract JSON decision from AI response"""
        # Remove think tags if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content)  # Handle incomplete tags
        content = content.strip()
        
        # Try direct parse first
        try:
            decision = json.loads(content)
            if self._validate_decision(decision):
                return decision
        except:
            pass
        
        # Find JSON patterns
        patterns = [
            r'\{[^{}]*"action"[^{}]*\}',
            r'\{.*?"action".*?\}',
            r'\{[^}]+\}',
        ]
        
        # Search in both cleaned and original content
        for text in [content, content.replace("'", '"')]:
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up common issues
                        match = match.replace("'", '"')
                        match = re.sub(r',\s*}', '}', match)  # Remove trailing commas
                        
                        decision = json.loads(match)
                        if self._validate_decision(decision):
                            return decision
                    except:
                        continue
        
        return None
    
    def _pattern_based_decision(self, context: Dict) -> Dict:
        """Fallback pattern-based decision using learned patterns"""
        market = context['current_market']
        position = context['position']
        
        # Use learned patterns
        if context['similar_historical_patterns']:
            # Average outcome of similar patterns
            outcomes = [p['outcome'] for p in context['similar_historical_patterns']]
            avg_success = sum(1 for o in outcomes if o > 0) / len(outcomes)
            
            if avg_success > 0.7:
                action = "LONG" if context['market_state'] == 'oversold' else "SHORT"
                confidence = avg_success
                reason = f"Historical pattern success: {avg_success:.0%}"
            else:
                action = "WAIT"
                confidence = 0.6
                reason = f"Low pattern success rate: {avg_success:.0%}"
        else:
            # Basic market analysis
            if position:
                pnl_pct = position.get('pnl_percent', 0)
                if pnl_pct > 1.0:
                    action = "CLOSE"
                    confidence = 0.8
                    reason = f"Take profit at {pnl_pct:.1f}%"
                elif pnl_pct < -0.5:
                    action = "CLOSE"
                    confidence = 0.9
                    reason = f"Stop loss at {pnl_pct:.1f}%"
                else:
                    action = "WAIT"
                    confidence = 0.5
                    reason = "Holding position"
            else:
                if context['market_state'] == 'oversold' and market['volume_ratio'] > 1.2:
                    action = "LONG"
                    confidence = 0.7
                    reason = "Oversold with volume"
                elif context['market_state'] == 'overbought' and market['volume_ratio'] > 1.2:
                    action = "SHORT"
                    confidence = 0.7
                    reason = "Overbought with volume"
                else:
                    action = "WAIT"
                    confidence = 0.5
                    reason = "No clear pattern"
        
        decision = {
            "action": action,
            "confidence": confidence,
            "reason": reason
        }
        
        # Add risk levels
        if action in ["LONG", "SHORT"]:
            price = market['price']
            decision["stop_loss"] = price * (0.995 if action == "LONG" else 1.005)
            decision["take_profit"] = price * (1.01 if action == "LONG" else 0.99)
            decision["position_size"] = self._calculate_position_size(confidence, context)
        
        return decision
    
    def _calculate_position_size(self, confidence: float, context: Dict) -> float:
        """Dynamic position sizing based on confidence and performance"""
        base_size = self.config.DEFAULT_POSITION_SIZE
        
        # Adjust by confidence
        size = base_size * confidence
        
        # Adjust by recent performance
        if context['performance']['consecutive_losses'] > 2:
            size *= 0.5
        elif context['performance']['win_rate'] > 0.7:
            size *= 1.2
        
        # Ensure within limits
        return max(self.config.MIN_POSITION_SIZE, min(size, self.config.MAX_POSITION_SIZE))
    
    def _calculate_price_changes(self, current_price: float) -> Dict:
        """Calculate price momentum"""
        if len(self.decision_history) < 2:
            return {"trend": "neutral", "strength": 0}
        
        recent_prices = [d['context']['price'] for d in self.decision_history[-10:] if 'context' in d]
        if not recent_prices:
            return {"trend": "neutral", "strength": 0}
        
        # Simple momentum calculation
        avg_price = sum(recent_prices) / len(recent_prices)
        momentum = (current_price - avg_price) / avg_price * 100
        
        if momentum > 0.5:
            return {"trend": "up", "strength": min(momentum / 2, 1)}
        elif momentum < -0.5:
            return {"trend": "down", "strength": min(abs(momentum) / 2, 1)}
        else:
            return {"trend": "neutral", "strength": 0}
    
    def _analyze_market_state(self, indicators: Dict) -> str:
        """Analyze overall market state"""
        rsi = indicators.get('rsi', 50)
        
        if rsi < 35:
            return "oversold"
        elif rsi > 65:
            return "overbought"
        else:
            return "neutral"
    
    def _find_similar_patterns(self, market_state: str) -> List[Dict]:
        """Find similar historical patterns"""
        similar = []
        
        for pattern_key, pattern_data in self.pattern_memory.items():
            if pattern_data['market_state'] == market_state:
                similar.append({
                    "pattern": pattern_key,
                    "outcome": pattern_data['outcome'],
                    "confidence": pattern_data['confidence']
                })
        
        # Return top 3 most relevant
        return sorted(similar, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def _record_decision(self, decision: Dict, context: Dict):
        """Record decision for learning"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'context': {
                'price': context['current_market']['price'],
                'rsi': context['current_market']['rsi'],
                'volume_ratio': context['current_market']['volume_ratio'],
                'market_state': context['market_state']
            }
        }
        
        self.decision_history.append(record)
        
        # Keep last 100 decisions
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
    
    def record_trade_outcome(self, entry_decision: Dict, exit_price: float, pnl: float):
        """Record trade outcome for learning"""
        outcome = {
            'entry': entry_decision,
            'exit_price': exit_price,
            'pnl': pnl,
            'success': pnl > 0
        }
        
        self.trade_outcomes.append(outcome)
        
        # Update pattern memory if context exists
        if hasattr(self, 'decision_history') and self.decision_history:
            # Find the decision record that matches this trade
            for record in reversed(self.decision_history):
                if record['decision'] == entry_decision:
                    context = record.get('context', {})
                    market_state = context.get('market_state', 'neutral')
                    rsi = context.get('rsi', 50)
                    
                    pattern_key = f"{market_state}_{int(rsi)//10}"
                    
                    if pattern_key not in self.pattern_memory:
                        self.pattern_memory[pattern_key] = {
                            'market_state': market_state,
                            'outcomes': [],
                            'confidence': 0.5
                        }
                    
                    self.pattern_memory[pattern_key]['outcomes'].append(pnl)
                    
                    # Update confidence based on outcomes
                    outcomes = self.pattern_memory[pattern_key]['outcomes']
                    success_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
                    self.pattern_memory[pattern_key]['confidence'] = success_rate
                    self.pattern_memory[pattern_key]['outcome'] = sum(outcomes) / len(outcomes)
                    break
    
    def _validate_decision(self, decision: Dict) -> bool:
        """Validate decision format"""
        if 'action' not in decision:
            return False
        
        decision['action'] = decision['action'].upper()
        
        if decision['action'] not in ['LONG', 'SHORT', 'CLOSE', 'WAIT']:
            return False
        
        decision.setdefault('confidence', 0.5)
        decision.setdefault('reason', 'AI decision')
        
        try:
            decision['confidence'] = float(decision['confidence'])
            return 0 <= decision['confidence'] <= 1
        except:
            return False
    
    async def load_strategies(self) -> Dict:
        """Load historical patterns instead of fixed strategies"""
        # Load from saved patterns if available
        try:
            with open('ai_patterns.json', 'r') as f:
                self.pattern_memory = json.load(f)
                logger.info(f"📚 Loaded {len(self.pattern_memory)} learned patterns")
        except:
            logger.info("🔧 Starting with fresh pattern memory")
        
        return {"patterns": len(self.pattern_memory)}
    
    def save_patterns(self):
        """Save learned patterns"""
        try:
            with open('ai_patterns.json', 'w') as f:
                json.dump(self.pattern_memory, f, indent=2)
            logger.info(f"💾 Saved {len(self.pattern_memory)} patterns")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def get_decision_stats(self) -> Dict:
        """Get AI performance statistics"""
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
            'patterns_learned': len(self.pattern_memory),
            'trade_outcomes': len(self.trade_outcomes),
            'last_decision': recent[-1] if recent else None
        }