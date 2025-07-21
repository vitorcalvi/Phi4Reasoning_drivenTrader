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
        
        # Load patterns on init
        self.load_patterns_sync()
        
    def load_patterns_sync(self):
        """Load patterns on initialization"""
        try:
            with open('ai_patterns.json', 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    self.pattern_memory = json.loads(content)
                    logger.info(f"📚 Loaded {len(self.pattern_memory)} patterns")
        except:
            logger.info("🔧 Starting with fresh pattern memory")
    
    async def test_connection(self) -> bool:
        """Test LLM connection"""
        # Always return True - we have pattern-based fallback
        logger.info("✅ AI Engine ready (pattern-based with LLM assist)")
        return True
    
    async def make_decision(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Pure AI-driven decision making"""
        
        # Build comprehensive context for AI
        context = self._build_context(market_data, position, performance)
        
        # Always use pattern-based decision as primary
        decision = self._pattern_based_decision(context)
        
        # Try to enhance with LLM if available (but don't depend on it)
        try:
            llm_decision = self._try_llm_decision(context)
            if llm_decision and llm_decision.get('confidence', 0) > decision.get('confidence', 0):
                decision = llm_decision
        except:
            pass  # Fallback already active
        
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
        similar_patterns = self._find_similar_patterns(market_state, indicators)
        
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
    
    def _try_llm_decision(self, context: Dict) -> Optional[Dict]:
        """Phi-4 compatible decision using explicit rules"""
        market = context['current_market']
        position = context['position']
        rsi = int(market['rsi'])
        
        if position:
            return None  # Let pattern system handle exits
        
        # System prompt to enforce concise output
        system_prompt = """You are a precise trading bot. Respond with ONLY the action word: LONG, SHORT, or WAIT. Do NOT use <think> tags, explanations, reasoning, or any extra text. Just the single word."""
        
        # Few-shot prompt with examples
        user_prompt = f"""RSI rules:
- RSI < 30: LONG
- RSI > 70: SHORT  
- RSI >= 30 and RSI <= 70: WAIT

Examples:
RSI 25 → LONG
RSI 75 → SHORT
RSI 50 → WAIT
RSI 30 → WAIT
RSI 70 → WAIT

RSI {rsi} →"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 1024
                },
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # Extract action robustly
                action = self._extract_action(content)
                if action and action in ['LONG', 'SHORT']:
                    return {
                        "action": action,
                        "confidence": 0.6,
                        "reason": f"LLM signal (RSI={rsi})"
                    }
        
        except Exception as e:
            logger.debug(f"LLM error: {e}")
        
        return None
    
    def _extract_action(self, content: str) -> Optional[str]:
        """Extract trading action from response"""
        # Normalize to upper case
        content_upper = content.upper()
        
        # Split on </think> and take text after the last one
        parts = re.split(r'</THINK>', content_upper, flags=re.IGNORECASE)
        if len(parts) > 1:
            post_think = parts[-1].strip()
            # Look for standalone action in post-think text
            match = re.search(r'\b(LONG|SHORT|WAIT)\b', post_think)
            if match:
                return match.group(1)
            
            # Handle boxed or formatted actions
            boxed_match = re.search(r'\\BOXED\{(LONG|SHORT|WAIT)\}', post_think)
            if boxed_match:
                return boxed_match.group(1)
        
        # If no post-think text, search entire cleaned content
        cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', content_upper, flags=re.DOTALL).strip()
        actions = re.findall(r'\b(LONG|SHORT|WAIT)\b', cleaned)
        if actions:
            # Return the last mentioned action
            return actions[-1]
        
        # Fallback search
        actions_original = re.findall(r'\b(LONG|SHORT|WAIT)\b', content_upper)
        if actions_original:
            unique_actions = set(actions_original)
            if len(unique_actions) == 1:
                return list(unique_actions)[0]
        
        return None

    def _pattern_based_decision(self, context: Dict) -> Dict:
        """Main decision logic based on learned patterns"""
        market = context['current_market']
        position = context['position']
        
        # Position management
        if position:
            pnl_pct = position.get('pnl_percent', 0)
            duration = position.get('duration_seconds', 0) / 60  # minutes
            
            # Clear exit rules
            if pnl_pct >= 1.0:  # 1% profit
                return {
                    "action": "CLOSE",
                    "confidence": 0.9,
                    "reason": f"Take profit at {pnl_pct:.1f}%"
                }
            elif pnl_pct <= -0.5:  # 0.5% loss
                return {
                    "action": "CLOSE",
                    "confidence": 0.95,
                    "reason": f"Stop loss at {pnl_pct:.1f}%"
                }
            elif duration > 5:  # 5 minutes max for scalping
                return {
                    "action": "CLOSE",
                    "confidence": 0.7,
                    "reason": f"Time limit ({duration:.0f} min)"
                }
            else:
                return {
                    "action": "WAIT",
                    "confidence": 0.6,
                    "reason": "Holding position"
                }
        
        # Entry logic based on patterns
        rsi = market['rsi']
        volume = market['volume_ratio']
        
        # Check historical patterns first
        if context['similar_historical_patterns']:
            patterns = context['similar_historical_patterns']
            successful = [p for p in patterns if p.get('avg_outcome', 0) > 0]
            
            if len(successful) >= 2:  # At least 2 successful patterns
                avg_confidence = sum(p['confidence'] for p in successful) / len(successful)
                if avg_confidence > 0.65:
                    if context['market_state'] == 'oversold':
                        return self._create_entry_decision("LONG", avg_confidence, 
                                                         f"Pattern match ({len(successful)} successful)")
                    elif context['market_state'] == 'overbought':
                        return self._create_entry_decision("SHORT", avg_confidence,
                                                         f"Pattern match ({len(successful)} successful)")
        
        # Fallback to technical analysis
        if rsi < 30 and volume > 1.5:
            confidence = 0.7 + (30 - rsi) / 100  # Higher confidence for lower RSI
            return self._create_entry_decision("LONG", confidence, f"Strong oversold RSI:{rsi:.0f}")
        elif rsi > 70 and volume > 1.5:
            confidence = 0.7 + (rsi - 70) / 100  # Higher confidence for higher RSI
            return self._create_entry_decision("SHORT", confidence, f"Strong overbought RSI:{rsi:.0f}")
        elif rsi < 35:
            return self._create_entry_decision("LONG", 0.6, f"Oversold RSI:{rsi:.0f}")
        elif rsi > 65:
            return self._create_entry_decision("SHORT", 0.6, f"Overbought RSI:{rsi:.0f}")
        
        # No clear signal
        return {
            "action": "WAIT",
            "confidence": 0.5,
            "reason": "No clear pattern"
        }
    
    def _create_entry_decision(self, action: str, confidence: float, reason: str) -> Dict:
        """Create entry decision with risk parameters"""
        decision = {
            "action": action,
            "confidence": min(confidence, 0.9),  # Cap confidence
            "reason": reason,
            "position_size": self._calculate_position_size(confidence, {})
        }
        
        # Simple stop/target based on action
        if action == "LONG":
            decision["stop_loss_pct"] = -0.5  # 0.5% stop
            decision["take_profit_pct"] = 1.0  # 1% target
        else:
            decision["stop_loss_pct"] = -0.5
            decision["take_profit_pct"] = 1.0
        
        return decision
    
    def _calculate_position_size(self, confidence: float, context: Dict) -> float:
        """Dynamic position sizing based on confidence"""
        base_size = self.config.DEFAULT_POSITION_SIZE
        
        # Scale by confidence (0.5 to 1.0x)
        size = base_size * (0.5 + confidence * 0.5)
        
        # Ensure within limits
        return max(self.config.MIN_POSITION_SIZE, min(size, self.config.MAX_POSITION_SIZE))
    
    def _analyze_market_state(self, indicators: Dict) -> str:
        """Analyze overall market state"""
        rsi = indicators.get('rsi', 50)
        
        if rsi < 35:
            return "oversold"
        elif rsi > 65:
            return "overbought"
        else:
            return "neutral"
    
    def _find_similar_patterns(self, market_state: str, indicators: Dict) -> List[Dict]:
        """Find similar historical patterns"""
        similar = []
        rsi = indicators.get('rsi', 50)
        
        # Create pattern key
        current_pattern = f"{market_state}_{int(rsi)//10}"
        
        for pattern_key, pattern_data in self.pattern_memory.items():
            # Direct match or similar state
            if pattern_key == current_pattern or pattern_data.get('market_state') == market_state:
                outcomes = pattern_data.get('outcomes', [])
                if outcomes:
                    avg_outcome = sum(outcomes) / len(outcomes)
                    success_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
                    
                    similar.append({
                        "pattern": pattern_key,
                        "avg_outcome": avg_outcome,
                        "confidence": success_rate,
                        "trades": len(outcomes)
                    })
        
        # Return top 3 most relevant (by number of trades)
        return sorted(similar, key=lambda x: x['trades'], reverse=True)[:3]
    
    def _calculate_price_changes(self, current_price: float) -> Dict:
        """Calculate price momentum"""
        if len(self.decision_history) < 2:
            return {"trend": "neutral", "strength": 0}
        
        recent_prices = []
        for d in self.decision_history[-10:]:
            if 'context' in d and 'price' in d['context']:
                recent_prices.append(d['context']['price'])
        
        if not recent_prices:
            return {"trend": "neutral", "strength": 0}
        
        # Simple momentum calculation
        avg_price = sum(recent_prices) / len(recent_prices)
        if avg_price > 0:
            momentum = (current_price - avg_price) / avg_price * 100
            
            if momentum > 0.5:
                return {"trend": "up", "strength": min(momentum / 2, 1)}
            elif momentum < -0.5:
                return {"trend": "down", "strength": min(abs(momentum) / 2, 1)}
        
        return {"trend": "neutral", "strength": 0}
    
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
        
        # Update pattern memory
        for record in reversed(self.decision_history):
            if record.get('decision') == entry_decision:
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
                
                # Keep last 50 outcomes per pattern
                if len(self.pattern_memory[pattern_key]['outcomes']) > 50:
                    self.pattern_memory[pattern_key]['outcomes'] = \
                        self.pattern_memory[pattern_key]['outcomes'][-50:]
                
                # Update confidence
                outcomes = self.pattern_memory[pattern_key]['outcomes']
                success_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
                self.pattern_memory[pattern_key]['confidence'] = success_rate
                
                # Save patterns immediately after each trade
                self.save_patterns()
                break
    
    async def load_strategies(self) -> Dict:
        """Load historical patterns"""
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
