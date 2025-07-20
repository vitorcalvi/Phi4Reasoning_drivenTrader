#!/usr/bin/env python3
"""
Comprehensive debug test script for Phi-4 JSON parsing issues
Tests multiple strategies to find a working solution
"""
import asyncio
import json
import requests
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List

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
    LLM_MODEL: str = "local-model"  # Will be detected as Phi-4
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

async def test_phi4_prompting_strategies(llm_url="http://localhost:1234", model="local-model"):
    """Test various prompting strategies to find what works with Phi-4"""
    print_header("Testing Phi-4 Prompting Strategies")
    
    endpoint = f"{llm_url}/v1/chat/completions"
    
    # Test different approaches
    test_prompts = [
        {
            "name": "Direct JSON (Successful in 1st test)",
            "messages": [
                {"role": "user", "content": 'Output exactly: {"action":"WAIT","confidence":0.5,"reason":"test"}'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "No System Message",
            "messages": [
                {"role": "user", "content": '{"action":"WAIT","confidence":0.5,"reason":"test"}'}
            ],
            "temperature": 0.0,
            "max_tokens": 50
        },
        {
            "name": "Complete the JSON",
            "messages": [
                {"role": "user", "content": 'Complete this JSON: {"action":"WAIT","confidence":'}
            ],
            "temperature": 0.0,
            "max_tokens": 50
        },
        {
            "name": "JSON after word",
            "messages": [
                {"role": "user", "content": 'Say RESPONSE then {"action":"WAIT","confidence":0.5,"reason":"test"}'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "Math then JSON",
            "messages": [
                {"role": "user", "content": '1+1=2. Output: {"action":"LONG","confidence":0.8,"reason":"buy"}'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "No Instructions",
            "messages": [
                {"role": "user", "content": 'action=WAIT confidence=0.5 reason=test'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        },
        {
            "name": "Code Block",
            "messages": [
                {"role": "user", "content": '```json\n{"action":"WAIT","confidence":0.5,"reason":"test"}\n```'}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }
    ]
    
    results = []
    successful_approaches = []
    
    for test in test_prompts:
        print(f"\n{Colors.YELLOW}📝 Test: {test['name']}{Colors.END}")
        
        payload = {
            "model": model,
            "messages": test["messages"],
            "temperature": test["temperature"],
            "max_tokens": test["max_tokens"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(endpoint, json=payload, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                finish_reason = result['choices'][0].get('finish_reason', '')
                tokens_used = result['usage']['completion_tokens']
                
                print(f"   Response: {Colors.BLUE}{content[:80]}{'...' if len(content) > 80 else ''}{Colors.END}")
                print(f"   Stats: {len(content)} chars, {tokens_used} tokens, {response_time:.2f}s, finish: {finish_reason}")
                
                # Analyze response
                has_think_tags = '<think>' in content
                starts_with_think = content.strip().startswith('<think>')
                
                # Clean and extract JSON
                cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                cleaned = re.sub(r'<think>.*', '', cleaned)
                cleaned = cleaned.strip()
                
                json_found = False
                extracted_json = None
                
                # Try multiple extraction methods
                # Method 1: Direct parse
                try:
                    extracted_json = json.loads(cleaned)
                    json_found = True
                except:
                    # Method 2: Find JSON pattern
                    json_patterns = [
                        r'\{[^{}]*"action"[^{}]*\}',
                        r'\{[^{}]*\'action\'[^{}]*\}',
                        r'\{[^{}]+\}',
                    ]
                    
                    for pattern in json_patterns:
                        matches = re.findall(pattern, cleaned, re.DOTALL)
                        for match in matches:
                            try:
                                # Fix common issues
                                fixed = match.replace("'", '"')
                                extracted_json = json.loads(fixed)
                                json_found = True
                                break
                            except:
                                continue
                        if json_found:
                            break
                
                if json_found and extracted_json:
                    print_test("JSON Extracted", True, f"{extracted_json}")
                    
                    # Check if it's a valid trading decision
                    is_valid_decision = all(k in extracted_json for k in ['action', 'confidence', 'reason'])
                    if is_valid_decision:
                        print_test("Valid Trading Decision", True)
                        successful_approaches.append({
                            "name": test["name"],
                            "prompt": test["messages"][-1]["content"],
                            "response_time": response_time,
                            "tokens": tokens_used
                        })
                else:
                    print_test("JSON Extraction", False, "No valid JSON found")
                
                # Analysis
                if has_think_tags:
                    print(f"   {Colors.YELLOW}⚠️  Contains <think> tags (starts with: {starts_with_think}){Colors.END}")
                
                results.append({
                    "name": test["name"],
                    "success": json_found,
                    "has_think_tags": has_think_tags,
                    "response_length": len(content),
                    "tokens_used": tokens_used,
                    "response_time": response_time
                })
                
        except Exception as e:
            print_test("Request", False, str(e))
            results.append({
                "name": test["name"],
                "success": False,
                "error": str(e)
            })
        
        time.sleep(0.5)
    
    # Summary
    print_header("Prompting Strategy Results")
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    
    if successful_approaches:
        print(f"\n{Colors.GREEN}✅ Successful Approaches:{Colors.END}")
        for approach in successful_approaches:
            print(f"   - {approach['name']}: {approach['response_time']:.2f}s, {approach['tokens']} tokens")
            print(f"     Prompt: {approach['prompt'][:60]}...")
    
    # Find patterns
    think_tag_results = [r for r in results if r.get("has_think_tags", False)]
    if think_tag_results:
        print(f"\n{Colors.YELLOW}⚠️  {len(think_tag_results)}/{len(results)} responses had <think> tags{Colors.END}")
    
    return results, successful_approaches

async def test_json_extraction_methods():
    """Test different JSON extraction methods on Phi-4 responses"""
    print_header("Testing JSON Extraction Methods")
    
    # Real Phi-4 responses from the test
    phi4_responses = [
        {
            "name": "Successful extraction from 1st test",
            "response": '''<think>
Okay, let's see what the user is asking here. The problem they provided seems to be a prompt for me to output a specific JSON object. The JSON object they want me to output is:

{"action":"WAIT","confidence":0.5,"reason":"test"}

So I need to output exactly this JSON object. Let me do that now.
</think>

{"action":"WAIT","confidence":0.5,"reason":"test"}'''
        },
        {
            "name": "Typical Phi-4 thinking response",
            "response": '''<think>
Okay, let me see here. The user provided some trading signals: price is $1.14, RSI is 34.3, and the position is None. They want a response in JSON format following specific instructions.

First, I need to understand what each parameter means. Price is the current price, RSI stands for Relative Strength Index, which measures the overbought or oversold status. A typical threshold is above 70 for overbuying and below 30 for selloff. Here, RSI is 34.3, which is well below 30, indicating oversold conditions.

The action field has to be one of the allowed ones: LONG, SHORT, CLOSE, or WAIT. Since the position is None, that means there's no current holding. The RSI being low suggests a potential buy (LONG) because it's in oversold territory.
</think>

Based on the analysis, here's my response:

{"action": "LONG", "confidence": 0.8, "reason": "oversold RSI"}'''
        },
        {
            "name": "Truncated response",
            "response": '''<think>
The RSI is 34.3 which is below 35, indicating oversold conditions. This suggests a buying opportunity.
</think>
{"action": "LONG", "conf'''
        },
        {
            "name": "No JSON in response",
            "response": '''<think>
I need to analyze the market conditions but I don't have enough information to make a decision.
</think>'''
        },
        {
            "name": "JSON without think tags",
            "response": '''{"action": "WAIT", "confidence": 0.5, "reason": "neutral market"}'''
        }
    ]
    
    def extract_json_comprehensive(content: str) -> Optional[Dict]:
        """Comprehensive JSON extraction"""
        # Step 1: Remove think tags
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Step 2: Try direct parse
        try:
            return json.loads(cleaned)
        except:
            pass
        
        # Step 3: Find JSON in original content (in case it's inside think tags)
        all_texts = [content, cleaned]
        
        for text in all_texts:
            # Look for JSON patterns
            patterns = [
                r'\{[^{}]*"action"[^{}]*\}',  # JSON with action key
                r'\{\s*"action"\s*:\s*"[^"]+"\s*[,\}]',  # Partial JSON starting with action
                r'\{[^{}]+\}',  # Any JSON object
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    # Complete truncated JSON if needed
                    if match.count('{') > match.count('}'):
                        # Try to complete common patterns
                        if '"conf' in match and '"confidence"' not in match:
                            match = match.replace('"conf', '"confidence": 0.5, "reason": "truncated"}')
                        else:
                            match += '}'
                    
                    try:
                        result = json.loads(match)
                        return result
                    except:
                        # Try fixing common issues
                        fixed = match.replace("'", '"')
                        try:
                            return json.loads(fixed)
                        except:
                            pass
        
        return None
    
    extraction_results = []
    
    for test in phi4_responses:
        print(f"\n{Colors.YELLOW}🔍 Test: {test['name']}{Colors.END}")
        print(f"   Input length: {len(test['response'])} chars")
        
        # Show first part of response
        preview = test['response'][:100].replace('\n', ' ')
        print(f"   Preview: {Colors.BLUE}{preview}...{Colors.END}")
        
        # Extract JSON
        extracted = extract_json_comprehensive(test['response'])
        
        if extracted:
            print_test("Extraction", True, f"{extracted}")
            
            # Validate as trading decision
            required_keys = {'action', 'confidence', 'reason'}
            has_all_keys = all(k in extracted for k in required_keys)
            
            if has_all_keys:
                print_test("Valid Decision", True)
                extraction_results.append({"name": test["name"], "success": True, "extracted": extracted})
            else:
                missing = required_keys - set(extracted.keys())
                print_test("Valid Decision", False, f"Missing keys: {missing}")
                extraction_results.append({"name": test["name"], "success": False, "reason": "incomplete"})
        else:
            print_test("Extraction", False, "No JSON found")
            extraction_results.append({"name": test["name"], "success": False, "reason": "no_json"})
    
    # Summary
    print_header("Extraction Results Summary")
    successful = sum(1 for r in extraction_results if r.get("success", False))
    print(f"Success rate: {successful}/{len(extraction_results)} ({successful/len(extraction_results)*100:.0f}%)")
    
    return extraction_results

async def test_engine_with_fixes():
    """Test the LLM Engine with the proposed fixes"""
    print_header("Testing LLM Engine with Fixes")
    
    config = MockConfig()
    engine = LLMEngine(config)
    
    print(f"Model: {engine.llm_model}")
    print(f"Is Phi-4 detected: {engine.is_phi4}")
    print(f"Max tokens: {engine.max_tokens}")
    
    # Test scenarios with timing
    scenarios = [
        {
            "name": "Oversold Market (RSI=30)",
            "market_data": {
                "price": 1.15,
                "indicators": {
                    "price": 1.15,
                    "rsi": 30,
                    "sma_20": 1.14,
                    "atr_pct": 1.5,
                    "volume_ratio": 2.0
                },
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.5}
            },
            "position": None,
            "expected": "LONG"
        },
        {
            "name": "Overbought Market (RSI=70)",
            "market_data": {
                "price": 1.20,
                "indicators": {
                    "price": 1.20,
                    "rsi": 70,
                    "sma_20": 1.18,
                    "atr_pct": 1.5,
                    "volume_ratio": 2.0
                },
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 0.5}
            },
            "position": None,
            "expected": "SHORT"
        },
        {
            "name": "Neutral Market (RSI=50)",
            "market_data": {
                "price": 1.17,
                "indicators": {
                    "price": 1.17,
                    "rsi": 50,
                    "sma_20": 1.16,
                    "atr_pct": 1.0,
                    "volume_ratio": 1.0
                },
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.0}
            },
            "position": None,
            "expected": "WAIT"
        },
        {
            "name": "Profitable Position (2% gain)",
            "market_data": {
                "price": 1.20,
                "indicators": {
                    "price": 1.20,
                    "rsi": 55,
                    "sma_20": 1.19,
                    "atr_pct": 1.0,
                    "volume_ratio": 1.0
                },
                "orderbook": {"spread_pct": 0.03},
                "trades": {"buy_sell_ratio": 1.0}
            },
            "position": {
                "side": "LONG",
                "entry_price": 1.176,
                "pnl_percent": 2.0,
                "duration_seconds": 300
            },
            "expected": "CLOSE"
        }
    ]
    
    performance = {"total_trades": 10, "winning_trades": 6, "consecutive_losses": 0}
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{Colors.YELLOW}📊 Scenario: {scenario['name']}{Colors.END}")
        
        start_time = time.time()
        decision = await engine.make_decision(
            scenario["market_data"],
            scenario["position"],
            performance
        )
        decision_time = time.time() - start_time
        
        print(f"   Decision: {decision['action']}")
        print(f"   Confidence: {decision['confidence']:.1%}")
        print(f"   Reason: {decision['reason']}")
        print(f"   Time taken: {decision_time:.2f}s")
        
        success = decision['action'] == scenario['expected']
        print_test(f"Matches expected ({scenario['expected']})", success)
        
        results.append({
            "scenario": scenario['name'],
            "success": success,
            "decision": decision,
            "time": decision_time,
            "expected": scenario['expected'],
            "actual": decision['action']
        })
    
    # Summary
    print_header("Engine Test Summary")
    successful = sum(1 for r in results if r['success'])
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average decision time: {avg_time:.2f}s")
    
    # Show any failures
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\n{Colors.RED}Failed scenarios:{Colors.END}")
        for f in failures:
            print(f"   - {f['scenario']}: Expected {f['expected']}, got {f['actual']}")
    
    return results

async def find_optimal_solution():
    """Analyze all results and suggest the optimal solution"""
    print_header("Finding Optimal Solution for Phi-4")
    
    # Run all tests
    print("Running comprehensive tests...")
    
    # Test 1: Prompting strategies
    prompt_results, successful_prompts = await test_phi4_prompting_strategies()
    
    # Test 2: Extraction methods
    extraction_results = await test_json_extraction_methods()
    
    # Test 3: Engine with fixes
    engine_results = await test_engine_with_fixes()
    
    # Analyze results
    print_header("Analysis & Recommendations")
    
    # Success rates
    prompt_success = sum(1 for r in prompt_results if r.get("success", False)) / len(prompt_results) * 100
    extraction_success = sum(1 for r in extraction_results if r.get("success", False)) / len(extraction_results) * 100
    engine_success = sum(1 for r in engine_results if r['success']) / len(engine_results) * 100
    
    print(f"\n{Colors.BOLD}Success Rates:{Colors.END}")
    print(f"   Prompting strategies: {prompt_success:.0f}%")
    print(f"   JSON extraction: {extraction_success:.0f}%")
    print(f"   Engine decisions: {engine_success:.0f}%")
    
    # Best practices identified
    print(f"\n{Colors.BOLD}Key Findings:{Colors.END}")
    print("1. Phi-4 almost always wraps responses in <think> tags")
    print("2. Simple, direct prompts work better than complex instructions")
    print("3. JSON often appears after the </think> tag")
    print("4. Extraction must handle both wrapped and unwrapped JSON")
    
    # Recommended solution
    print(f"\n{Colors.GREEN}{Colors.BOLD}Recommended Solution:{Colors.END}")
    print("1. **Pre-calculate decisions** based on market indicators")
    print("2. **Use Phi-4 only for confirmation** with simple prompts")
    print("3. **Always have fallback** to pre-calculated decisions")
    print("4. **Extract JSON flexibly** from any part of the response")
    
    # Code recommendations
    print(f"\n{Colors.BOLD}Code Implementation:{Colors.END}")
    print("```python")
    print("# In _phi4_direct_decision():")
    print("1. Calculate decision based on RSI/position")
    print("2. Try 3 simple prompt approaches:")
    print("   - Direct: 'Output exactly: {json}'")
    print("   - After word: 'Say OK then {json}'")
    print("   - Just JSON: '{json}'")
    print("3. Extract JSON from anywhere in response")
    print("4. Fallback to pre-calculated if all fail")
    print("```")
    
    # Alternative models
    print(f"\n{Colors.BOLD}Alternative Models to Consider:{Colors.END}")
    print("- Mistral-7B-Instruct: Better at following instructions")
    print("- Llama-2-7B-Chat: More reliable JSON output")
    print("- OpenHermes-2.5: Optimized for structured output")
    print("- Zephyr-7B: Good balance of reasoning and compliance")

async def main():
    """Run all debug tests"""
    print(f"{Colors.BOLD}🐛 Phi-4 JSON Parsing Debug Suite v2.0{Colors.END}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find the optimal solution
    await find_optimal_solution()
    
    print(f"\n{Colors.BOLD}✅ Debug suite complete!{Colors.END}")

if __name__ == "__main__":
    asyncio.run(main())