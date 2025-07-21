#!/usr/bin/env python3
"""
Streamlined Phi-4 implementation using explicit if-then rules with improved prompting
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
        
        # First try LLM with improved prompt, retry once if fails
        llm_result = self._try_llm_signal(rsi_int)
        if llm_result:
            return llm_result
        
        llm_result = self._try_llm_signal(rsi_int, retry=True)  # Retry with alternative prompt
        if llm_result:
            return llm_result
        
        # Fallback to deterministic rules if LLM fails
        print(f"DEBUG: LLM failed for RSI {rsi_int}, using fallback rules")
        return self._get_deterministic_signal(rsi_int)
    
    def _try_llm_signal(self, rsi_int: int, retry: bool = False) -> Optional[str]:
        """Try to get signal from LLM with few-shot prompting"""
        if retry:
            # Alternative prompt for retry: More explicit with boundaries
            prompt = f"""You are a trading bot following exact RSI rules. Respond with ONLY the action word.

Rules (strict boundaries):
- If RSI strictly less than 30 (RSI < 30): LONG
- If RSI strictly greater than 70 (RSI > 70): SHORT
- Otherwise (RSI >= 30 and RSI <= 70): WAIT

Examples:
RSI: 29 → LONG
RSI: 30 → WAIT
RSI: 70 → WAIT
RSI: 71 → SHORT

Current RSI: {rsi_int}

Your answer (one word only):"""
        else:
            # Primary prompt with few-shot examples
            prompt = f"""You are a trading bot. Follow these RSI rules exactly and respond with ONLY the action: LONG, SHORT, or WAIT. No explanations, no thinking tags, no extra text.

Rules:
- RSI < 30: LONG
- RSI > 70: SHORT  
- RSI >= 30 and <= 70: WAIT

Examples:
Input: RSI 25 → LONG
Input: RSI 75 → SHORT
Input: RSI 50 → WAIT
Input: RSI 30 → WAIT
Input: RSI 70 → WAIT

Input: RSI {rsi_int} →"""
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 200  # Increased to allow for completion
                },
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                print(f"DEBUG: Raw LLM response for RSI {rsi_int} (retry={retry}): {content}")
                action = self._extract_action(content)
                if action:
                    return action
                else:
                    print(f"DEBUG: Extraction failed on {'retry' if retry else 'first try'}")
            
        except Exception as e:
            print(f"LLM error: {e}")
        
        return None
    
    def _get_deterministic_signal(self, rsi: int) -> str:
        """Fallback deterministic rules - always works"""
        if rsi < 30:
            return "LONG"
        elif rsi > 70:
            return "SHORT"
        else:
            return "WAIT"
    
    def _extract_action(self, content: str) -> Optional[str]:
        """Extract trading action from response - robust logic"""
        # Clean content thoroughly
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>.*?</[^>]+>', '', content, flags=re.DOTALL)
        content = content.replace('\\boxed{', '').replace('}', '').replace('"', '').replace('→', '').strip()
        
        # Normalize to upper case
        content_upper = content.upper()
        
        # Look for exact standalone action
        match = re.match(r'^(LONG|SHORT|WAIT)$', content_upper.strip())
        if match:
            return match.group(1)
        
        # Look for pattern like "LONG" anywhere, but prioritize the first clear one
        actions = re.findall(r'\b(LONG|SHORT|WAIT)\b', content_upper)
        if actions:
            # Return the last one, as it might be the final answer
            return actions[-1]
        
        print(f"DEBUG: No action extracted from: {content}")
        return None
    
    def test_rules(self):
        """Test the explicit rules with known cases"""
        test_cases = [
            (25, "LONG"),   # Below 30
            (75, "SHORT"),  # Above 70
            (50, "WAIT"),   # Between 30-70
            (30, "WAIT"),   # Boundary
            (70, "WAIT"),   # Boundary
            (29, "LONG"),   # Just below 30
            (71, "SHORT"),  # Just above 70
            (15, "LONG"),   # Well below 30
            (85, "SHORT"),  # Well above 70
            (45, "WAIT"),   # Middle range
        ]
        
        print("Testing Explicit Rules:")
        print("-" * 50)
        
        correct = 0
        total = len(test_cases)
        
        for rsi, expected in test_cases:
            result = self.get_trading_signal(rsi)
            result_str = result if result else "NONE"
            status = "✅" if result == expected else "❌"
            if result == expected:
                correct += 1
            print(f"RSI {rsi:2d}: {result_str:5s} (expected {expected:5s}) {status}")
        
        print("-" * 50)
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


# Alternative: Pure deterministic version (no LLM)
class DeterministicRules:
    def get_trading_signal(self, rsi: float) -> str:
        """Pure rule-based signal - no LLM dependency"""
        rsi_int = int(rsi)
        if rsi_int < 30:
            return "LONG"
        elif rsi_int > 70:
            return "SHORT"
        else:
            return "WAIT"


# Drop-in replacement for existing LLM decision method
def _try_llm_decision(context: Dict, llm_url: str, llm_model: str, use_deterministic_fallback: bool = True) -> Optional[Dict]:
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
    
    if use_deterministic_fallback:
        # Use hybrid approach (LLM with fallback)
        phi4 = Phi4ExplicitRules(llm_url, llm_model)
    else:
        # Use pure deterministic rules (no LLM)
        phi4 = DeterministicRules()
    
    action = phi4.get_trading_signal(rsi)
    
    if action and action in ['LONG', 'SHORT']:
        method = "LLM+fallback" if use_deterministic_fallback else "deterministic"
        return {
            "action": action,
            "confidence": 0.6,
            "reason": f"{method} rules (RSI={int(rsi)})"
        }
    
    return None


# Test the implementation
if __name__ == "__main__":
    print("=== Testing Hybrid LLM + Fallback ===")
    phi4_hybrid = Phi4ExplicitRules()
    phi4_hybrid.test_rules()
    
    print("\n=== Testing Pure Deterministic ===")
    phi4_deterministic = DeterministicRules()
    test_cases = [(25, "LONG"), (75, "SHORT"), (50, "WAIT"), (30, "WAIT"), (70, "WAIT")]
    
    print("Testing Pure Deterministic Rules:")
    print("-" * 50)
    for rsi, expected in test_cases:
        result = phi4_deterministic.get_trading_signal(rsi)
        status = "✅" if result == expected else "❌"
        print(f"RSI {rsi:2d}: {result:5s} (expected {expected:5s}) {status}")
