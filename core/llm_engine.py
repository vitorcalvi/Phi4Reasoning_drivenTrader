"""
LLM Engine - The brain of the trading bot
Analyzes market data and makes all trading decisions
Uses LM Studio for local LLM inference
"""
import json
import logging
import yaml
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config):
        self.config = config
        # LM Studio configuration
        self.llm_url = config.LLM_URL  # Default: http://localhost:1234
        self.llm_model = config.LLM_MODEL  # e.g., "microsoft/phi-4", "mistral", etc.
        self.strategies = {}
        self.current_strategy = None
        self.decision_history = []
        
        # LM Studio specific settings
        self.lm_studio_endpoint = f"{self.llm_url}/v1/chat/completions"
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        
    async def test_connection(self) -> bool:
        """Test LM Studio connection"""
        try:
            # Test LM Studio API endpoint
            test_payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "Test connection"},
                    {"role": "user", "content": "Hello"}
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
            else:
                logger.warning(f"⚠️ LM Studio returned status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ LM Studio not running at {self.llm_url}")
            logger.info("💡 Start LM Studio and load a model to enable AI decisions")
        except Exception as e:
            logger.warning(f"⚠️ LM Studio connection error: {e}")
            
        logger.info("📊 Will use fallback decision logic")
        return True  # Continue even if LLM is offline
    
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
        
        # Set default strategy
        if 'scalping' in self.strategies:
            self.current_strategy = 'scalping'
        elif self.strategies:
            self.current_strategy = list(self.strategies.keys())[0]
        
        return self.strategies
    
    async def make_decision(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Main decision-making function"""
        
        # Build comprehensive context
        context = self._build_context(market_data, position, performance)
        
        # Try LLM decision first
        decision = await self._llm_decision(context)
        
        # If LLM fails, use fallback logic
        if not decision:
            decision = self._fallback_decision(context)
        
        # Record decision
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'market_price': market_data.get('price', 0),
            'position': position is not None
        })
        
        # Keep only recent history
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return decision
    
    def _build_context(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Build comprehensive context for decision making"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'market': market_data,
            'position': position,
            'performance': performance,
            'strategy': self.strategies.get(self.current_strategy, {}),
            'recent_decisions': self._prepare_for_json(self.decision_history[-5:]) if self.decision_history else []
        }
        
        # Add derived insights
        indicators = market_data.get('indicators', {})
        
        # Market condition analysis
        context['market_condition'] = self._analyze_market_condition(market_data)
        
        # Risk assessment
        context['risk_level'] = self._assess_risk_level(market_data, position, performance)
        
        return context
    
    def _analyze_market_condition(self, market_data: Dict) -> Dict:
        """Analyze overall market condition"""
        indicators = market_data.get('indicators', {})
        orderbook = market_data.get('orderbook', {})
        trades = market_data.get('trades', {})
        
        condition = {
            'trend': 'neutral',
            'volatility': 'normal',
            'momentum': 'neutral',
            'volume': 'normal',
            'sentiment': 'neutral'
        }
        
        # Trend analysis
        price = indicators.get('price', 0)
        sma_20 = indicators.get('sma_20', price)
        
        if price > sma_20 * 1.01:
            condition['trend'] = 'bullish'
        elif price < sma_20 * 0.99:
            condition['trend'] = 'bearish'
        
        # Volatility
        atr_pct = indicators.get('atr_pct', 0)
        if atr_pct > 2:
            condition['volatility'] = 'high'
        elif atr_pct < 0.5:
            condition['volatility'] = 'low'
        
        # Momentum
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        
        if rsi > 65 and macd_hist > 0:
            condition['momentum'] = 'bullish'
        elif rsi < 35 and macd_hist < 0:
            condition['momentum'] = 'bearish'
        
        # Volume
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            condition['volume'] = 'high'
        elif volume_ratio < 0.5:
            condition['volume'] = 'low'
        
        # Sentiment from order flow
        buy_sell_ratio = trades.get('buy_sell_ratio', 1)
        if buy_sell_ratio > 1.5:
            condition['sentiment'] = 'bullish'
        elif buy_sell_ratio < 0.7:
            condition['sentiment'] = 'bearish'
        
        return condition
    
    def _assess_risk_level(self, market_data: Dict, position: Optional[Dict], performance: Dict) -> Dict:
        """Assess current risk level"""
        risk = {
            'overall': 'medium',
            'factors': []
        }
        
        # Market risk factors
        spread_pct = market_data.get('orderbook', {}).get('spread_pct', 0)
        if spread_pct > 0.1:
            risk['factors'].append('high_spread')
        
        volatility = market_data.get('indicators', {}).get('atr_pct', 0)
        if volatility > 3:
            risk['factors'].append('high_volatility')
        
        # Performance risk factors
        consecutive_losses = performance.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            risk['factors'].append('consecutive_losses')
            risk['overall'] = 'high'
        
        # Position risk
        if position:
            pnl_pct = position.get('pnl_percent', 0)
            if pnl_pct < -1:
                risk['factors'].append('position_underwater')
            
            duration = position.get('duration_seconds', 0) / 60  # minutes
            if duration > 10:
                risk['factors'].append('position_too_long')
        
        # Determine overall risk
        if len(risk['factors']) >= 3:
            risk['overall'] = 'high'
        elif len(risk['factors']) == 0:
            risk['overall'] = 'low'
        
        return risk
    
    async def _llm_decision(self, context: Dict) -> Optional[Dict]:
        """Get decision from LM Studio"""
        try:
            # Create comprehensive prompt
            prompt = self._create_llm_prompt(context)
            
            # Prepare LM Studio request
            request_payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert crypto trading AI. Analyze market data and make trading decisions.
                        Always respond with valid JSON only. Consider risk management and market conditions carefully.
                        Your goal is consistent profits while minimizing drawdowns."""
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
            
            # Call LM Studio
            response = requests.post(
                self.lm_studio_endpoint,
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content from LM Studio response
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    # Log response length for debugging
                    logger.debug(f"🧠 LM Studio response length: {len(content)} chars")
                    
                    # Extract JSON from response
                    # Try to find JSON in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        
                        try:
                            decision = json.loads(json_str)
                            
                            # Validate decision
                            if self._validate_decision(decision):
                                logger.info(f"🧠 LLM Decision: {decision['action']} - {decision.get('reason', 'No reason')}")
                                logger.info(f"🔮 Confidence: {decision.get('confidence', 0):.1%}")
                                return decision
                            else:
                                logger.warning("⚠️ Invalid decision format from LLM")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse LLM JSON response: {e}")
                            logger.debug(f"📄 Response content: {json_str[:500]}...")
                    else:
                        logger.warning("⚠️ No JSON found in LLM response")
                        logger.debug(f"📄 Full response: {content[:500]}...")
                else:
                    logger.error("❌ Unexpected LM Studio response format")
                    
            else:
                logger.error(f"❌ LM Studio request failed with status: {response.status_code}")
                if response.text:
                    logger.debug(f"Error details: {response.text}")
                    
        except requests.exceptions.Timeout:
            logger.error("❌ LM Studio request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Could not connect to LM Studio")
        except Exception as e:
            logger.error(f"❌ LLM decision error: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _prepare_for_json(self, obj):
        """Convert non-serializable objects to JSON-compatible format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def _create_llm_prompt(self, context: Dict) -> str:
        """Create detailed prompt for LLM with emphasis on JSON output"""
        market = context['market']
        position = context['position']
        performance = context['performance']
        strategy = context['strategy']
        market_condition = context['market_condition']
        risk_level = context['risk_level']
        
        # Convert datetime objects to strings for JSON serialization
        position_json = self._prepare_for_json(position) if position else None
        performance_json = self._prepare_for_json(performance)
        
        # Add JSON emphasis for better LM Studio compatibility
        # Try a simpler prompt format if the model is struggling
        simple_mode = os.getenv('SIMPLE_PROMPTS', 'False').lower() == 'true'
        
        if simple_mode:
            # Simplified prompt for models that struggle with complex formatting
            prompt = f"""Trading bot needs a decision. Current price: ${market.get('price', 0):.2f}

RSI: {market.get('indicators', {}).get('rsi', 50):.1f}
Volume ratio: {market.get('indicators', {}).get('volume_ratio', 1):.1f}
Position: {"None" if not position else position['side']}

Respond with this JSON format only:
{{
    "action": "WAIT",
    "confidence": 0.7,
    "reason": "Your reason here",
    "position_size": 10
}}

Action must be: LONG, SHORT, CLOSE, or WAIT"""
        else:
            # Full detailed prompt
            prompt = f"""
CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no text before or after.
Start your response with {{ and end with }}

TRADING DECISION REQUIRED

CURRENT MARKET DATA:
- Price: ${market.get('price', 0):.2f}
- Indicators: {json.dumps(market.get('indicators', {}), indent=2)}
- Order Book: {json.dumps(market.get('orderbook', {}), indent=2)}
- Recent Trades: {json.dumps(market.get('trades', {}), indent=2)}

MARKET CONDITION ANALYSIS:
{json.dumps(market_condition, indent=2)}

CURRENT POSITION:
{json.dumps(position_json, indent=2) if position_json else "No position"}

PERFORMANCE METRICS:
{json.dumps(performance_json, indent=2)}

RISK ASSESSMENT:
{json.dumps(risk_level, indent=2)}

ACTIVE STRATEGY ({self.current_strategy}):
{json.dumps(strategy, indent=2)}

REQUIRED JSON FORMAT (respond with this structure ONLY):
{{
    "action": "LONG" or "SHORT" or "CLOSE" or "WAIT" or "UPDATE_SL" or "UPDATE_TP",
    "position_size": number (USDT amount, only for LONG/SHORT),
    "stop_loss": number (price level),
    "take_profit": number (price level),
    "confidence": number (0.0 to 1.0),
    "reason": "Brief explanation of decision",
    "risk_score": number (1-10, where 10 is highest risk),
    "expected_duration": "scalp" or "short" or "medium",
    "strategy_adjustments": {{
        "key": "new_value"
    }}
}}

Consider:
1. Current market conditions and trend
2. Risk/reward ratio
3. Position sizing based on confidence
4. Recent performance and consecutive losses
5. Spread costs and liquidity
6. Strategy rules and parameters

REMEMBER: Output ONLY the JSON object, nothing else!
"""
        
        return prompt
    
    def _validate_decision(self, decision: Dict) -> bool:
        """Validate LLM decision"""
        # Fix common field name variations
        if 'action' not in decision:
            # Check for common variations
            if 'decision' in decision:
                decision['action'] = decision['decision']
            elif 'trade' in decision:
                decision['action'] = decision['trade']
            elif 'signal' in decision:
                decision['action'] = decision['signal']
        
        # Ensure confidence exists
        if 'confidence' not in decision:
            # Check for variations
            if 'conf' in decision:
                decision['confidence'] = decision['conf']
            elif 'probability' in decision:
                decision['confidence'] = decision['probability']
            elif 'certainty' in decision:
                decision['confidence'] = decision['certainty']
            else:
                # Default confidence if missing
                decision['confidence'] = 0.5
        
        # Ensure reason exists
        if 'reason' not in decision:
            if 'reasoning' in decision:
                decision['reason'] = decision['reasoning']
            elif 'explanation' in decision:
                decision['reason'] = decision['explanation']
            elif 'rationale' in decision:
                decision['reason'] = decision['rationale']
            else:
                decision['reason'] = 'No reason provided'
        
        required_fields = ['action', 'confidence', 'reason']
        
        # Check required fields
        for field in required_fields:
            if field not in decision:
                logger.debug(f"Missing required field: {field}")
                return False
        
        # Normalize action to uppercase
        decision['action'] = decision['action'].upper()
        
        # Validate action
        valid_actions = ['LONG', 'SHORT', 'CLOSE', 'WAIT', 'UPDATE_SL', 'UPDATE_TP']
        if decision['action'] not in valid_actions:
            logger.debug(f"Invalid action: {decision['action']}")
            return False
        
        # Validate confidence
        confidence = decision.get('confidence', 0)
        if not (0 <= confidence <= 1):
            logger.debug(f"Invalid confidence: {confidence}")
            return False
        
        # Validate position size for entry actions
        if decision['action'] in ['LONG', 'SHORT']:
            if 'position_size' not in decision:
                decision['position_size'] = self.config.DEFAULT_POSITION_SIZE
        
        return True
    
    def _fallback_decision(self, context: Dict) -> Dict:
        """Fallback decision logic when LLM is unavailable"""
        logger.info("📊 Using fallback decision logic")
        
        market = context['market']
        position = context['position']
        performance = context['performance']
        risk_level = context['risk_level']
        
        indicators = market.get('indicators', {})
        
        # Default decision
        decision = {
            'action': 'WAIT',
            'confidence': 0.5,
            'reason': 'No clear signal',
            'risk_score': 5
        }
        
        # If we have a position, check exit conditions
        if position:
            pnl_pct = position.get('pnl_percent', 0)
            duration = position.get('duration_seconds', 0) / 60
            
            # Exit conditions
            if pnl_pct > 1:  # 1% profit
                decision = {
                    'action': 'CLOSE',
                    'confidence': 0.8,
                    'reason': 'Take profit target reached',
                    'risk_score': 2
                }
            elif pnl_pct < -0.5:  # 0.5% loss
                decision = {
                    'action': 'CLOSE',
                    'confidence': 0.9,
                    'reason': 'Stop loss triggered',
                    'risk_score': 8
                }
            elif duration > 5:  # 5 minutes
                decision = {
                    'action': 'CLOSE',
                    'confidence': 0.7,
                    'reason': 'Time stop - position too long',
                    'risk_score': 6
                }
        
        # Entry conditions (only if no position and low risk)
        elif risk_level['overall'] != 'high':
            rsi = indicators.get('rsi', 50)
            volume_ratio = indicators.get('volume_ratio', 1)
            spread_pct = market.get('orderbook', {}).get('spread_pct', 0.05)
            
            # Long signal
            if rsi < 35 and volume_ratio > 1.5 and spread_pct < 0.05:
                decision = {
                    'action': 'LONG',
                    'position_size': self.config.DEFAULT_POSITION_SIZE,
                    'stop_loss': indicators['price'] * 0.995,
                    'take_profit': indicators['price'] * 1.01,
                    'confidence': 0.7,
                    'reason': 'Oversold with volume spike',
                    'risk_score': 4
                }
            
            # Short signal
            elif rsi > 65 and volume_ratio > 1.5 and spread_pct < 0.05:
                decision = {
                    'action': 'SHORT',
                    'position_size': self.config.DEFAULT_POSITION_SIZE,
                    'stop_loss': indicators['price'] * 1.005,
                    'take_profit': indicators['price'] * 0.99,
                    'confidence': 0.7,
                    'reason': 'Overbought with volume spike',
                    'risk_score': 4
                }
        
        # High risk - wait
        else:
            decision['reason'] = f"High risk: {', '.join(risk_level['factors'])}"
            decision['risk_score'] = 9
        
        return decision
    
    def update_strategy(self, strategy_name: str):
        """Switch to a different strategy"""
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            logger.info(f"📝 Switched to strategy: {strategy_name}")
        else:
            logger.warning(f"⚠️ Strategy not found: {strategy_name}")
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling various formats"""
        if not content:
            return None
        
        # Clean the content first
        content = content.strip()
        
        # Method 1: Try to parse the entire response as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Look for JSON between curly braces
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON errors
                # Remove trailing commas
                import re
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Try to extract JSON from markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Method 4: Look for JSON after common prefixes
        prefixes = [
            "Here is the JSON:",
            "Here's the JSON:",
            "JSON:",
            "Response:",
            "Output:",
            "Decision:",
            "```json",
            "```"
        ]
        
        for prefix in prefixes:
            if prefix in content:
                start = content.find(prefix) + len(prefix)
                json_part = content[start:].strip()
                
                # Try to find JSON in this part
                json_start = json_part.find('{')
                json_end = json_part.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    try:
                        return json.loads(json_part[json_start:json_end])
                    except json.JSONDecodeError:
                        continue
        
        logger.warning("Could not extract valid JSON from LLM response")
        logger.debug(f"Response content: {content[:200]}...")
        return None
    
    def get_decision_stats(self) -> Dict:
        """Get statistics about recent decisions"""
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