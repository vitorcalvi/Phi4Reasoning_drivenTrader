#!/usr/bin/env python3
"""
Streamlined Phi-4 implementation using explicit if-then rules
"""
import requests
import re
import json
from typing import Dict, Optional


class Phi4ExplicitRules:
    def __init__(self, llm_url="http://localhost:1234/v1/chat/completions", model="local-model"):
        self.llm_url = llm_url
        self.model = model
    
    def get_trading_signal(self, rsi: float) -> Optional[str]:
        """
        Get trading signal using explicit if-then rules that Phi-4 can follow
        """
        rsi_int = int(rsi)
        
        # Improved prompt: Force concise response with no explanations
        prompt = f"""Rules:
1. If RSI below 30: LONG
2. If RSI above 70: SHORT  
3. If RSI between 30-70: WAIT

RSI = {rsi_int}

Respond ONLY with the action: LONG, SHORT, or WAIT. No explanations or additional text.

Answer:"""
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 100  # Increased to avoid truncation
                },
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                print(f"DEBUG: Raw LLM response for RSI {rsi_int}: {content}")  # Added for debugging
                return self._extract_action(content)
            
        except Exception as e:
            print(f"LLM error: {e}")
        
        return None
    
    def _extract_action(self, content: str) -> Optional[str]:
        """Extract trading action from response"""
        # Clean content: Remove <think> blocks and common wrappers
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<\w+>.*?</\w+>', '', content, flags=re.DOTALL)  # Remove other potential tags
        content = content.replace('\\boxed{', '').replace('}', '').replace('"', '').strip().upper()
        
        # Improved extraction: Look for patterns like "LONG" or "The action is LONG"
        match = re.search(r'\b(LONG|SHORT|WAIT)\b', content)
        if match:
            return match.group(1)
        
        # Fallback: If no exact match, check for substrings (less reliable, but helpful for verbose models)
        content_lower = content.lower()
        if 'long' in content_lower:
            return 'LONG'
        elif 'short' in content_lower:
            return 'SHORT'
        elif 'wait' in content_lower:
            return 'WAIT'
        
        print(f"DEBUG: No action extracted from: {content}")  # Added for debugging
        return None
    
    def test_rules(self):
        """Test the explicit rules with known cases"""
        test_cases = [
            (25, "LONG"),   # Below 30
            (75, "SHORT"),  # Above 70
            (50, "WAIT"),   # Between 30-70
            (30, "WAIT"),   # Boundary
            (70, "WAIT"),   # Boundary
            (29, "LONG"),   # Additional: Just below 30
            (71, "SHORT"),  # Additional: Just above 70
        ]
        
        print("Testing Explicit Rules:")
        print("-" * 40)
        
        for rsi, expected in test_cases:
            result = self.get_trading_signal(rsi)
            result_str = result if result else "NONE"
            status = "✅" if result == expected else "❌"
            print(f"RSI {rsi:2d}: {result_str:5s} (expected {expected}) {status}")


# Drop-in replacement for existing LLM decision method
def _try_llm_decision(context: Dict, llm_url: str, llm_model: str) -> Optional[Dict]:
    """
    Phi-4 compatible decision using explicit rules
    Drop-in replacement for existing LLM methods
    """
    market = context['current_market']
    position = context['position']
    rsi = float(market['rsi'])
    
    # Skip if we have a position (let pattern system handle exits)
    if position:
        return None
    
    phi4 = Phi4ExplicitRules(llm_url, llm_model)
    action = phi4.get_trading_signal(rsi)
    
    if action and action in ['LONG', 'SHORT']:
        return {
            "action": action,
            "confidence": 0.6,
            "reason": f"LLM explicit rules (RSI={int(rsi)})"
        }
    
    return None


# Test the implementation
if __name__ == "__main__":
    phi4 = Phi4ExplicitRules()
    phi4.test_rules()
