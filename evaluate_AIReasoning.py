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
        
        # First try LLM with primary prompt
        llm_result = self._try_llm_signal(rsi_int, variant=0)
        if llm_result:
            return llm_result
        
        # Retry with alternative prompt
        llm_result = self._try_llm_signal(rsi_int, variant=1)
        if llm_result:
            return llm_result
        
        # Second retry with another variant
        llm_result = self._try_llm_signal(rsi_int, variant=2)
        if llm_result:
            return llm_result
        
        # Fallback to deterministic rules if all LLM attempts fail
        print(f"DEBUG: All LLM attempts failed for RSI {rsi_int}, using fallback rules")
        return self._get_deterministic_signal(rsi_int)
    
    def _try_llm_signal(self, rsi_int: int, variant: int = 0) -> Optional[str]:
        """Try to get signal from LLM with different prompt variants"""
        system_prompt = """You are a precise trading bot. Respond with ONLY the action word: LONG, SHORT, or WAIT. Do NOT use <think> tags, explanations, reasoning, or any extra text. Just the single word."""
        
        if variant == 0:
            # Primary: Few-shot with arrow format
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

RSI {rsi_int} →"""
        
        elif variant == 1:
            # Variant 1: Explicit strict boundaries with output instruction
            user_prompt = f"""Strict RSI rules:
- If RSI < 30: LONG
- If RSI > 70: SHORT
- If RSI >= 30 and RSI <= 70: WAIT

Examples:
RSI: 29 → LONG
RSI: 30 → WAIT
RSI: 70 → WAIT
RSI: 71 → SHORT

Current RSI: {rsi_int}

Output ONLY the action word:"""
        
        elif variant == 2:
            # Variant 2: Force concise output with placeholder
            user_prompt = f"""Determine action for RSI {rsi_int}:
- < 30: LONG
- > 70: SHORT
- else: WAIT

Action:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 1024  # Increased significantly to prevent truncation
                },
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                print(f"DEBUG: Raw LLM response for RSI {rsi_int} (variant={variant}): {content}")
                action = self._extract_action(content)
                if action:
                    return action
                else:
                    print(f"DEBUG: Extraction failed for variant {variant}")
            
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
        """Extract trading action from response - improved to handle tags and truncation"""
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
        
        # If no post-think text (truncated inside think), or no match, search entire cleaned content
        cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', content_upper, flags=re.DOTALL).strip()
        actions = re.findall(r'\b(LONG|SHORT|WAIT)\b', cleaned)
        if actions:
            # Return the last mentioned action (likely the conclusion)
            return actions[-1]
        
        # Fallback search in original
        actions_original = re.findall(r'\b(LONG|SHORT|WAIT)\b', content_upper)
        if actions_original:
            # Avoid picking from rules; prefer if there's only one unique, but to be safe, return None if ambiguous
            unique_actions = set(actions_original)
            if len(unique_actions) == 1:
                return list(unique_actions)[0]
        
        print(f"DEBUG: No reliable action extracted from: {content}")
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
