#!/usr/bin/env python3
"""
Debug why LLM gives wrong answers for RSI=75 and times out for RSI=50
"""
import requests
import re
import time

def debug_llm_responses():
    """Debug the actual LLM responses"""
    print("🔍 Debugging LLM Responses")
    print("=" * 60)
    
    llm_url = "http://localhost:1234/v1/chat/completions"
    
    # Test with different token limits and prompts
    test_configs = [
        {
            "name": "Original prompt",
            "prompt": "RSI: {rsi}, Volume: {volume}x. Should I go LONG, SHORT, or WAIT?",
            "max_tokens": 1000
        },
        {
            "name": "Clear rules",
            "prompt": "Trading rules: RSI<30=LONG, RSI>70=SHORT, else=WAIT. Current RSI: {rsi}, Volume: {volume}x. Decision?",
            "max_tokens": 1000
        },
        {
            "name": "Direct instruction",
            "prompt": "RSI={rsi}. If RSI<30 reply LONG. If RSI>70 reply SHORT. Otherwise reply WAIT. What is your reply?",
            "max_tokens": 500
        },
        {
            "name": "Teaching example",
            "prompt": "Example: RSI=25→LONG, RSI=75→SHORT, RSI=50→WAIT. Now, RSI={rsi}. Your answer?",
            "max_tokens": 500
        }
    ]
    
    test_cases = [
        {"rsi": 25, "volume": 1.5, "expected": "LONG"},
        {"rsi": 75, "volume": 2.0, "expected": "SHORT"},
        {"rsi": 50, "volume": 1.0, "expected": "WAIT"}
    ]
    
    for config in test_configs:
        print(f"\n📋 Testing: {config['name']}")
        print("-" * 40)
        
        all_correct = True
        
        for test in test_cases:
            prompt = config['prompt'].format(rsi=test['rsi'], volume=test['volume'])
            
            try:
                response = requests.post(
                    llm_url,
                    json={
                        "model": "local-model",
                        "messages": [
                            {"role": "system", "content": "You are a trading bot. Follow the rules exactly."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0,
                        "max_tokens": config['max_tokens']
                    },
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    tokens = data['usage']['completion_tokens']
                    finish = data['choices'][0].get('finish_reason', 'unknown')
                    
                    # Extract the answer
                    # Remove think tags
                    clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    
                    # Find action words
                    found_action = None
                    for action in ['LONG', 'SHORT', 'WAIT']:
                        if action in clean_content.upper():
                            found_action = action
                            break
                    
                    # Check if correct
                    is_correct = found_action == test['expected']
                    if not is_correct:
                        all_correct = False
                    
                    print(f"RSI={test['rsi']}: {found_action or 'NONE'} {'✅' if is_correct else '❌'} " +
                          f"({tokens} tokens, {finish})")
                    
                    # If wrong, show why
                    if not is_correct and found_action:
                        print(f"  ⚠️  LLM thinks RSI={test['rsi']} → {found_action}")
                        if test['rsi'] == 75 and found_action == 'LONG':
                            print(f"  💡 Model confused: treating high RSI as bullish signal")
                        
                    # Show reasoning if available
                    if not is_correct and '<think>' in content:
                        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        if think_match:
                            reasoning = think_match.group(1).strip()
                            # Find key phrases
                            if 'overbought' in reasoning.lower():
                                print(f"  📝 Model mentions 'overbought' but still says {found_action}")
                            if 'oversold' in reasoning.lower():
                                print(f"  📝 Model mentions 'oversold'")
                else:
                    print(f"RSI={test['rsi']}: Request failed")
                    all_correct = False
                    
            except Exception as e:
                print(f"RSI={test['rsi']}: Error - {str(e)[:50]}")
                all_correct = False
            
            time.sleep(0.5)
        
        if all_correct:
            print(f"\n✅ This prompt format works correctly!")
            return config
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS")
    print("=" * 60)
    print("\n❌ The model has a fundamental misunderstanding:")
    print("   - It treats high RSI (75) as a LONG signal")
    print("   - This is backwards - high RSI should be SHORT")
    print("\n💡 The issue is the model's training, not the prompt!")
    
    return None

def test_simple_fix():
    """Test if we can fix it with very clear instructions"""
    print("\n\n🔧 Testing Simple Fix")
    print("=" * 60)
    
    llm_url = "http://localhost:1234/v1/chat/completions"
    
    # Very explicit prompt
    prompt = """You must follow these exact rules:
1. If RSI is 75, you MUST respond: SHORT
2. If RSI is 25, you MUST respond: LONG  
3. If RSI is 50, you MUST respond: WAIT

Current RSI is 75. What is your response?"""
    
    try:
        response = requests.post(
            llm_url,
            json={
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            },
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print(f"Response: {content}")
            
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            if 'SHORT' in clean.upper():
                print("\n✅ Model can follow explicit rules!")
                print("💡 Solution: Use very explicit if-then rules")
            else:
                print("\n❌ Model still confused even with explicit rules")
                print("💡 This model may have incorrect RSI understanding baked in")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    working_config = debug_llm_responses()
    test_simple_fix()