#!/usr/bin/env python3
"""
Debug test script for Phi-4 JSON parsing issues
Tests different prompting strategies and JSON extraction
"""
import asyncio
import json
import requests
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

# Try to import from the correct location
try:
    from core.llm_engine import LLMEngine
except ImportError:
    try:
        from llm_engine import LLMEngine
    except ImportError:
        print("❌ Could not import LLMEngine. Make sure llm_engine.py is in the same directory or in core/")
        exit(1)

# Mock config
@dataclass
class MockConfig:
    LLM_URL: str = "http://localhost:1234"
    LLM_MODEL: str = "local-model"  # Will be detected as Phi-4 if it contains 'phi'
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 500
    DEFAULT_POSITION_SIZE: float = 100.0

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")

def print_test(name, result, details=""):
    if result:
        print(f"{Colors.GREEN}✅ {name}{Colors.END}")
    else:
        print(f"{Colors.RED}❌ {name}{Colors.END}")
    if details:
        print(f"   {Colors.BLUE}{details}{Colors.END}")

async def test_phi4_response_patterns(llm_url="http://localhost:1234", model="local-model"):
    """Test how Phi-4 responds to different prompt patterns"""
    print_header("Testing Phi-4 Response Patterns")
    
    endpoint = f"{llm_url}/v1/chat/completions"
    
    test_prompts = [
        {
            "name": "Direct JSON Output",
            "messages": [
                {"role": "user", "content": 'Output exactly: {"action":"WAIT","confidence":0.5,"reason":"test"}'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "System Instruction",
            "messages": [
                {"role": "system", "content": "Output only JSON. No other text."},
                {"role": "user", "content": "action=WAIT, confidence=0.5, reason=test"}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "Copy Task",
            "messages": [
                {"role": "system", "content": '{"action":"WAIT","confidence":0.5,"reason":"test"}'},
                {"role": "user", "content": "Copy the system message exactly."}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "One Word Response",
            "messages": [
                {"role": "user", "content": "Say only: WAIT"}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        },
        {
            "name": "JSON Template",
            "messages": [
                {"role": "system", "content": "Fill template: {\"action\":\"[A]\",\"confidence\":[C],\"reason\":\"[R]\"}"},
                {"role": "user", "content": "A=WAIT, C=0.5, R=test"}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
    ]
    
    results = []
    
    for test in test_prompts:
        print(f"\n{Colors.YELLOW}📝 Test: {test['name']}{Colors.END}")
        
        payload = {
            "model": model,
            "messages": test["messages"],
            "temperature": test["temperature"],
            "max_tokens": test["max_tokens"]
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                finish_reason = result['choices'][0].get('finish_reason', '')
                
                print(f"   Response: {Colors.BLUE}{content[:100]}{'...' if len(content) > 100 else ''}{Colors.END}")
                print(f"   Length: {len(content)} chars, Finish: {finish_reason}")
                
                # Check if it's valid JSON
                json_found = False
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        json_found = True
                        print_test("JSON Extracted", True, f"{parsed}")
                    except:
                        print_test("JSON Extraction", False, "Found JSON pattern but couldn't parse")
                else:
                    print_test("JSON Extraction", False, "No JSON pattern found")
                
                # Check for think tags
                if '<think>' in content:
                    print(f"   {Colors.YELLOW}⚠️  Contains <think> tags{Colors.END}")
                
                results.append({
                    "name": test["name"],
                    "success": json_found,
                    "response_length": len(content),
                    "has_think_tags": '<think>' in content
                })
                
        except Exception as e:
            print_test("Request", False, str(e))
            results.append({
                "name": test["name"],
                "success": False,
                "error": str(e)
            })
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print_header("Results Summary")
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    
    best_approach = max(results, key=lambda x: (x.get("success", False), -x.get("response_length", 999)))
    if best_approach.get("success"):
        print(f"{Colors.GREEN}Best approach: {best_approach['name']}{Colors.END}")
    else:
        print(f"{Colors.RED}No fully successful approach found{Colors.END}")
    
    return results

async def test_llm_engine_decisions():
    """Test the LLMEngine with various market conditions"""
    print_header("Testing LLM Engine Decisions")
    
    config = MockConfig()
    engine = LLMEngine(config)
    
    # Check if model is detected as Phi-4
    print(f"Model: {engine.llm_model}")
    print(f"Is Phi-4: {engine.is_phi4}")
    
    # Test connection
    connected = await engine.test_connection()
    print_test("LM Studio Connection", connected)
    
    if not connected:
        print(f"{Colors.YELLOW}⚠️  LM Studio not connected, will use fallback logic{Colors.END}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Oversold (RSI=25)",
            "market": {
                "price": 50000,
                "indicators": {"price": 50000, "rsi": 25, "sma_20": 49500, "atr_pct": 1.5, "volume_ratio": 2.0},
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.5}
            },
            "position": None,
            "expected": "LONG"
        },
        {
            "name": "Overbought (RSI=75)",
            "market": {
                "price": 50000,
                "indicators": {"price": 50000, "rsi": 75, "sma_20": 49500, "atr_pct": 1.5, "volume_ratio": 2.0},
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 0.5}
            },
            "position": None,
            "expected": "SHORT"
        },
        {
            "name": "Neutral (RSI=50)",
            "market": {
                "price": 50000,
                "indicators": {"price": 50000, "rsi": 50, "sma_20": 49500, "atr_pct": 1.5, "volume_ratio": 1.0},
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.0}
            },
            "position": None,
            "expected": "WAIT"
        },
        {
            "name": "Profitable Position",
            "market": {
                "price": 50000,
                "indicators": {"price": 50000, "rsi": 50, "sma_20": 49500, "atr_pct": 1.5, "volume_ratio": 1.0},
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.0}
            },
            "position": {"side": "LONG", "entry_price": 49000, "pnl_percent": 2.0, "duration_seconds": 300},
            "expected": "CLOSE"
        }
    ]
    
    performance = {"total_trades": 10, "winning_trades": 6, "consecutive_losses": 0}
    
    for scenario in test_scenarios:
        print(f"\n{Colors.YELLOW}📊 Scenario: {scenario['name']}{Colors.END}")
        
        decision = await engine.make_decision(
            scenario["market"],
            scenario["position"],
            performance
        )
        
        print(f"   Decision: {decision['action']}")
        print(f"   Confidence: {decision['confidence']:.1%}")
        print(f"   Reason: {decision['reason']}")
        
        print_test(
            f"Expected {scenario['expected']}", 
            decision['action'] == scenario['expected'],
            f"Got {decision['action']}"
        )

async def test_json_extraction():
    """Test JSON extraction from various Phi-4 response formats"""
    print_header("Testing JSON Extraction Methods")
    
    config = MockConfig()
    engine = LLMEngine(config)
    
    test_responses = [
        {
            "name": "Clean JSON",
            "response": '{"action":"WAIT","confidence":0.5,"reason":"test"}'
        },
        {
            "name": "JSON with think tags",
            "response": '<think>Analyzing...</think>{"action":"LONG","confidence":0.8,"reason":"oversold"}'
        },
        {
            "name": "JSON with prefix text",
            "response": 'Okay, let me analyze. {"action":"SHORT","confidence":0.7,"reason":"overbought"}'
        },
        {
            "name": "Truncated think tag",
            "response": '<think>The RSI is low so{"action":"LONG","confidence":0.9,"reason":"buy signal"}'
        },
        {
            "name": "Multiple JSON objects",
            "response": 'First: {"x":1} then {"action":"WAIT","confidence":0.5,"reason":"neutral"}'
        },
        {
            "name": "No JSON",
            "response": 'I cannot provide a JSON response at this time.'
        }
    ]
    
    for test in test_responses:
        print(f"\n{Colors.YELLOW}🔍 Test: {test['name']}{Colors.END}")
        print(f"   Input: {Colors.BLUE}{test['response'][:60]}{'...' if len(test['response']) > 60 else ''}{Colors.END}")
        
        # Clean response
        cleaned = engine._clean_response(test['response'])
        if cleaned != test['response']:
            print(f"   Cleaned: {cleaned[:60]}{'...' if len(cleaned) > 60 else ''}")
        
        # Extract JSON
        extracted = engine._extract_json(cleaned)
        
        if extracted:
            print_test("Extraction", True, f"{extracted}")
            
            # Validate if it's a trading decision
            if engine._validate_decision(extracted.copy()):
                print_test("Validation", True, "Valid trading decision")
            else:
                print_test("Validation", False, "Not a valid trading decision")
        else:
            print_test("Extraction", False, "No JSON found")

async def test_phi4_direct_decision():
    """Test the new _phi4_direct_decision method"""
    print_header("Testing Phi-4 Direct Decision Method")
    
    config = MockConfig()
    config.LLM_MODEL = "microsoft/phi-4"  # Force Phi-4 detection
    engine = LLMEngine(config)
    
    print(f"Is Phi-4: {engine.is_phi4}")
    
    test_contexts = [
        {
            "name": "Oversold Market",
            "market": {"indicators": {"price": 50000, "rsi": 30}},
            "position": None,
            "expected_action": "LONG"
        },
        {
            "name": "Overbought Market",
            "market": {"indicators": {"price": 50000, "rsi": 70}},
            "position": None,
            "expected_action": "SHORT"
        },
        {
            "name": "Neutral Market",
            "market": {"indicators": {"price": 50000, "rsi": 50}},
            "position": None,
            "expected_action": "WAIT"
        },
        {
            "name": "Profitable Position",
            "market": {"indicators": {"price": 50000, "rsi": 50}},
            "position": {"pnl_percent": 1.5},
            "expected_action": "CLOSE"
        }
    ]
    
    for test in test_contexts:
        print(f"\n{Colors.YELLOW}🧪 Test: {test['name']}{Colors.END}")
        
        context = {
            "market": test["market"],
            "position": test["position"],
            "performance": {},
            "strategy": {},
            "market_condition": {},
            "risk_level": {"overall": "medium"}
        }
        
        # Test direct decision method
        decision = await engine._phi4_direct_decision(context)
        
        if decision:
            print_test("Decision Generated", True)
            print(f"   Action: {decision['action']}")
            print(f"   Confidence: {decision['confidence']}")
            print(f"   Reason: {decision['reason']}")
            
            print_test(
                f"Expected {test['expected_action']}", 
                decision['action'] == test['expected_action']
            )
        else:
            print_test("Decision Generated", False, "Failed to generate decision")

async def main():
    """Run all debug tests"""
    print(f"{Colors.BOLD}🐛 Phi-4 JSON Parsing Debug Suite{Colors.END}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Raw Phi-4 responses
    await test_phi4_response_patterns()
    
    # Test 2: LLM Engine decisions
    await test_llm_engine_decisions()
    
    # Test 3: JSON extraction
    await test_json_extraction()
    
    # Test 4: Direct decision method
    await test_phi4_direct_decision()
    
    print_header("Debug Complete")
    print("\n💡 Recommendations:")
    print("1. If Phi-4 keeps adding <think> tags, use the direct decision method")
    print("2. Keep prompts ultra-simple - just ask for exact output")
    print("3. Use temperature=0.0 for consistent results")
    print("4. Pre-calculate decisions and ask Phi-4 to output the JSON")
    print("5. Always have a fallback to pre-calculated decisions")

if __name__ == "__main__":
    asyncio.run(main())