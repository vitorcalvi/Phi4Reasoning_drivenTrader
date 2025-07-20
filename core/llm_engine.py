"""
LLM Engine - The brain of the trading bot
Analyzes market data and makes all trading decisions
Uses LM Studio for local LLM inference
"""
import json
import logging
import yaml
import requests
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.llm_url = config.LLM_URL  
        self.llm_model = config.LLM_MODEL  
        self.strategies = {}
        self.current_strategy = None
        self.decision_history = []
        
        self.lm_studio_endpoint = f"{self.llm_url}/v1/chat/completions"
        self.temperature = 0.3  # Increased from 0.1 for better responses
        self.max_tokens = 500  # Increased to handle Phi-4's reasoning style
        
        # Check if model is Phi-4 for special handling
        self.is_phi4 = 'phi' in self.llm_model.lower() or 'local-model' in self.llm_model.lower()
        
    async def test_connection(self) -> bool:
        """Test LM Studio connection"""
        try:
            test_payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": "Reply with: OK"}
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }
            
            response = requests.post(
                self.lm_studio_endpoint,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Connected to LM Studio at {self.llm_url}")
                logger.info(f"🤖 Using model: {self.llm_model}")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ LM Studio connection error: {e}")
            
        logger.info("📊 Will use fallback decision logic")
        return True
    
    async def load_strategies(self) -> Dict:
        """Load trading strategies from YAML files"""
        strategies_dir = Path("strategies")
        
        for strategy_file in strategies_dir.glob("*.yaml"):
            try:
                with open(strategy_file, 'r') as f:
                    strategy = yaml.safe_load(f)
                    strategy_name = strategy.get('name', strategy_file.stem)
                    self.strategies[strategy_name] = strategy
                    logger.info(f"📚 Loaded strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"❌ Error loading {strategy_file}: {e}")
        
        if 'scalping' in self.strategies:
            self.current_strategy = 'scalping'
        elif self.strategies:
            self.current_strategy = list(self.strategies.keys())[0]
        
        return self.strategies
    
    async def make_decision(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Main decision-making function"""
        context = self._build_context(market_data, position, performance)
        
        decision = await self._llm_decision(context)
        
        if not decision:
            decision = self._fallback_decision(context)
        
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'market_price': market_data.get('price', 0),
            'position': position is not None
        })
        
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return decision
    
    def _build_context(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Build context for decision making"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'market': market_data,
            'position': position,
            'performance': performance,
            'strategy': self.strategies.get(self.current_strategy, {}),
            'recent_decisions': self.decision_history[-5:] if self.decision_history else []
        }
        
        context['market_condition'] = self._analyze_market_condition(market_data)
        context['risk_level'] = self._assess_risk_level(market_data, position, performance)
        
        return context
    
    def _analyze_market_condition(self, market_data: Dict) -> Dict:
        """Analyze market condition"""
        indicators = market_data.get('indicators', {})
        
        condition = {
            'trend': 'neutral',
            'volatility': 'normal',
            'momentum': 'neutral',
            'volume': 'normal'
        }
        
        price = indicators.get('price', 0)
        sma_20 = indicators.get('sma_20', price)
        
        if price > sma_20 * 1.01:
            condition['trend'] = 'bullish'
        elif price < sma_20 * 0.99:
            condition['trend'] = 'bearish'
        
        atr_pct = indicators.get('atr_pct', 0)
        if atr_pct > 2:
            condition['volatility'] = 'high'
        elif atr_pct < 0.5:
            condition['volatility'] = 'low'
        
        rsi = indicators.get('rsi', 50)
        if rsi > 65:
            condition['momentum'] = 'bullish'
        elif rsi < 35:
            condition['momentum'] = 'bearish'
        
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            condition['volume'] = 'high'
        elif volume_ratio < 0.5:
            condition['volume'] = 'low'
        
        return condition
    
    def _assess_risk_level(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Assess risk level"""
        risk = {
            'overall': 'medium',
            'factors': []
        }
        
        spread_pct = market_data.get('orderbook', {}).get('spread_pct', 0)
        if spread_pct > 0.1:
            risk['factors'].append('high_spread')
        
        volatility = market_data.get('indicators', {}).get('atr_pct', 0)
        if volatility > 3:
            risk['factors'].append('high_volatility')
        
        consecutive_losses = performance.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            risk['factors'].append('consecutive_losses')
            risk['overall'] = 'high'
        
        if position:
            pnl_pct = position.get('pnl_percent', 0)
            if pnl_pct < -1:
                risk['factors'].append('position_underwater')
            
            duration = position.get('duration_seconds', 0) / 60
            if duration > 10:
                risk['factors'].append('position_too_long')
        
        if len(risk['factors']) >= 3:
            risk['overall'] = 'high'
        elif len(risk['factors']) == 0:
            risk['overall'] = 'low'
        
        return risk
    
    async def _llm_decision(self, context: Dict) -> Optional[Dict]:
        """Get decision from LM Studio"""
        try:
            # For Phi-4, bypass the thinking problem entirely
            if self.is_phi4:
                return self._phi4_direct_decision(context)
            
            # Normal flow for other models
            prompt = self._create_llm_prompt(context)
            
            request_payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Reply ONLY with JSON in format: {\"action\":\"WAIT\",\"confidence\":0.5,\"reason\":\"waiting\"}"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(
                self.lm_studio_endpoint,
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    # Clean the response
                    content = self._clean_response(content)
                    
                    # Extract JSON
                    decision = self._extract_json(content)
                    
                    if decision and self._validate_decision(decision):
                        logger.info(f"🧠 LLM Decision: {decision['action']} - {decision.get('reason', 'No reason')}")
                        logger.info(f"🔮 Confidence: {decision.get('confidence', 0):.1%}")
                        return decision
                    else:
                        logger.warning(f"⚠️ Failed to extract valid JSON from: {content[:200]}...")
                        
        except Exception as e:
            logger.error(f"❌ LLM decision error: {e}")
        
        return None
    
    def _phi4_direct_decision(self, context: Dict) -> Optional[Dict]:
        """Direct decision for Phi-4 without complex prompting"""
        try:
            indicators = context['market'].get('indicators', {})
            position = context['position']
            rsi = indicators.get('rsi', 50)
            
            # Make decision based on RSI
            if position:
                # Have position - check exit
                pnl = position.get('pnl_percent', 0)
                if pnl > 1.0:
                    action, confidence, reason = "CLOSE", 0.8, "profit target"
                elif pnl < -0.5:
                    action, confidence, reason = "CLOSE", 0.9, "stop loss"
                else:
                    action, confidence, reason = "WAIT", 0.6, "holding"
            else:
                # No position - check entry
                if rsi < 35:
                    action, confidence, reason = "LONG", 0.7, f"oversold RSI {rsi:.0f}"
                elif rsi > 65:
                    action, confidence, reason = "SHORT", 0.7, f"overbought RSI {rsi:.0f}"
                else:
                    action, confidence, reason = "WAIT", 0.5, f"neutral RSI {rsi:.0f}"
            
            # Build the decision
            decision = {
                "action": action,
                "confidence": confidence,
                "reason": reason
            }
            
            # For Phi-4, try a very simple prompt that might work
            json_str = json.dumps(decision)
            
            # Try 3 different approaches
            attempts = [
                {
                    # Approach 1: Direct output
                    "messages": [
                        {"role": "user", "content": f"Output exactly: {json_str}"}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100
                },
                {
                    # Approach 2: Single word then JSON
                    "messages": [
                        {"role": "user", "content": f"Say OK then output: {json_str}"}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 150
                },
                {
                    # Approach 3: Just the JSON
                    "messages": [
                        {"role": "user", "content": json_str}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 50
                }
            ]
            
            for i, attempt in enumerate(attempts):
                request_payload = {
                    "model": self.llm_model,
                    "messages": attempt["messages"],
                    "temperature": attempt["temperature"],
                    "max_tokens": attempt["max_tokens"],
                    "stream": False
                }
                
                try:
                    response = requests.post(
                        self.lm_studio_endpoint,
                        json=request_payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            
                            # Clean any think tags
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                            content = re.sub(r'<think>.*', '', content)
                            content = content.strip()
                            
                            # Try to extract JSON
                            extracted = self._extract_json(content)
                            if extracted and self._validate_decision(extracted):
                                logger.info(f"✅ Phi-4 approach {i+1} succeeded")
                                return extracted
                            
                except Exception as e:
                    logger.debug(f"Phi-4 attempt {i+1} failed: {e}")
                    continue
            
            # If all LLM attempts fail, return our pre-calculated decision
            logger.info(f"📊 Using pre-calculated decision for Phi-4")
            if self._validate_decision(decision):
                return decision
                
        except Exception as e:
            logger.error(f"❌ Phi-4 direct decision error: {e}")
        
        return None
    
    def _clean_response(self, content: str) -> str:
        """Clean LLM response"""
        # Remove think tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content)
        
        # For Phi-4, also remove everything before first {
        if self.is_phi4 and '{' in content:
            content = content[content.find('{'):]
        
        # Remove ellipsis
        content = content.replace('...', '').replace('…', '')
        
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        return content.strip()
    
    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extract JSON from cleaned response"""
        if not content:
            return None
        
        # Method 1: Direct parse
        try:
            return json.loads(content)
        except:
            pass
        
        # Method 2: Find JSON object with action key
        # Look for patterns like {"action": ... }
        json_patterns = [
            r'\{[^{}]*"action"[^{}]*\}',  # JSON with "action" key
            r'\{[^{}]*\'action\'[^{}]*\}',  # JSON with 'action' key
            r'\{\s*"action"\s*:\s*"[^"]+"\s*,\s*"confidence"\s*:\s*[0-9.]+\s*,\s*"reason"\s*:\s*"[^"]+"\s*\}',  # Full pattern
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    # Fix common issues
                    fixed = match.replace("'", '"')  # Replace single quotes
                    return json.loads(fixed)
                except:
                    continue
        
        # Method 3: Find any JSON object
        match = re.search(r'\{[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        # Method 4: Find nested JSON
        start = content.find('{')
        if start != -1:
            brace_count = 0
            for i in range(start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(content[start:i+1])
                        except:
                            pass
                        break
        
        return None
    
    def _create_llm_prompt(self, context: Dict) -> str:
        """Create simple prompt for LLM"""
        market = context['market']
        position = context['position']
        indicators = market.get('indicators', {})
        
        prompt = f"""Price: ${indicators.get('price', 0):.2f}
RSI: {indicators.get('rsi', 50):.1f}
Position: {position['side'] if position else 'None'}

Respond with this JSON format only:
{{"action": "WAIT", "confidence": 0.5, "reason": "waiting"}}

Rules: action must be LONG/SHORT/CLOSE/WAIT
Output JSON only, no other text."""
        
        return prompt
    
    def _validate_decision(self, decision: Dict) -> bool:
        """Validate and fix decision"""
        # Fix field names
        field_mapping = {
            'action': ['decision', 'Decision', 'trade', 'signal'],
            'confidence': ['conf', 'probability', 'certainty'],
            'reason': ['reasoning', 'Reasoning', 'explanation', 'rationale']
        }
        
        for target, alternatives in field_mapping.items():
            if target not in decision:
                for alt in alternatives:
                    if alt in decision:
                        decision[target] = decision[alt]
                        break
        
        # Convert action values
        action_mapping = {'BUY': 'LONG', 'SELL': 'SHORT'}
        if 'action' in decision:
            decision['action'] = action_mapping.get(decision['action'], decision['action']).upper()
        
        # Set defaults
        if 'confidence' not in decision:
            decision['confidence'] = 0.5
        if 'reason' not in decision:
            decision['reason'] = 'No reason provided'
        
        # Validate
        if 'action' not in decision:
            return False
        
        valid_actions = ['LONG', 'SHORT', 'CLOSE', 'WAIT', 'UPDATE_SL', 'UPDATE_TP']
        if decision['action'] not in valid_actions:
            return False
        
        confidence = float(decision.get('confidence', 0))
        if not (0 <= confidence <= 1):
            return False
        
        decision['confidence'] = confidence
        
        if decision['action'] in ['LONG', 'SHORT'] and 'position_size' not in decision:
            decision['position_size'] = self.config.DEFAULT_POSITION_SIZE
        
        return True
    
    def _fallback_decision(self, context: Dict) -> Dict:
        """Simple fallback decision logic"""
        logger.info("📊 Using fallback decision logic")
        
        market = context['market']
        position = context['position']
        indicators = market.get('indicators', {})
        risk_level = context['risk_level']
        
        decision = {
            'action': 'WAIT',
            'confidence': 0.5,
            'reason': 'No clear signal'
        }
        
        if position:
            pnl_pct = position.get('pnl_percent', 0)
            
            if pnl_pct > 1:
                decision = {
                    'action': 'CLOSE',
                    'confidence': 0.8,
                    'reason': 'Take profit'
                }
            elif pnl_pct < -0.5:
                decision = {
                    'action': 'CLOSE',
                    'confidence': 0.9,
                    'reason': 'Stop loss'
                }
        else:
            rsi = indicators.get('rsi', 50)
            spread_pct = market.get('orderbook', {}).get('spread_pct', 0.05)
            
            # Entry conditions (only if low risk and good spread)
            if risk_level['overall'] != 'high' and spread_pct < 0.05:
                if rsi < 35:
                    decision = {
                        'action': 'LONG',
                        'position_size': self.config.DEFAULT_POSITION_SIZE,
                        'stop_loss': indicators['price'] * 0.995,
                        'take_profit': indicators['price'] * 1.01,
                        'confidence': 0.7,
                        'reason': 'Oversold',
                        'risk_score': 4
                    }
                elif rsi > 65:
                    decision = {
                        'action': 'SHORT',
                        'position_size': self.config.DEFAULT_POSITION_SIZE,
                        'stop_loss': indicators['price'] * 1.005,
                        'take_profit': indicators['price'] * 0.99,
                        'confidence': 0.7,
                        'reason': 'Overbought',
                        'risk_score': 4
                    }
        
        return decision
    
    def update_strategy(self, strategy_name: str):
        """Switch strategy"""
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            logger.info(f"📝 Switched to strategy: {strategy_name}")
    
    def get_decision_stats(self) -> Dict:
        """Get decision statistics"""
        if not self.decision_history:
            return {}
        
        recent = self.decision_history[-20:]
        
        action_counts = {}
        total_confidence = 0
        
        for decision in recent:
            action = decision['decision']['action']
            action_counts[action] = action_counts.get(action, 0) + 1
            total_confidence += decision['decision'].get('confidence', 0)
        
        return {
            'total_decisions': len(recent),
            'action_distribution': action_counts,
            'avg_confidence': total_confidence / len(recent) if recent else 0,
            'last_decision': recent[-1] if recent else None
        }