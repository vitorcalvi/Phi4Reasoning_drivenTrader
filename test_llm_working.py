#!/usr/bin/env python3
"""Test if the LLM fix is working"""
import asyncio
from config import Config
from core.llm_engine import LLMEngine

async def test_llm():
    config = Config()
    llm = LLMEngine(config)
    
    test_cases = [
        {"rsi": 25, "volume": 1.5, "expected": "LONG"},
        {"rsi": 75, "volume": 2.0, "expected": "SHORT"},
        {"rsi": 50, "volume": 1.0, "expected": "WAIT"}
    ]
    
    print("Testing LLM with explicit rules...")
    print("=" * 40)
    
    for test in test_cases:
        # Build context
        context = {
            "current_market": {
                "rsi": test["rsi"],
                "volume_ratio": test["volume"]
            },
            "position": None
        }
        
        # Test
        decision = llm._try_llm_decision(context)
        
        if decision:
            action = decision.get("action", "NONE")
            correct = action == test["expected"]
            print(f"RSI={test['rsi']}: {action} {'✅' if correct else '❌'}")
        else:
            print(f"RSI={test['rsi']}: No decision ❌")
    
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(test_llm())
