#!/usr/bin/env python3
"""
Test the fixed LLM extraction
"""
import requests
import re
import time

def test_llm_extraction():
    """Test if LLM now works properly"""
    print("🧪 Testing Fixed LLM Extraction")
    print("=" * 60)
    
    llm_url = "http://localhost:1234/v1/chat/completions"
    
    test_cases = [
        {"rsi": 25, "volume": 1.5, "expected": "LONG"},
        {"rsi": 75, "volume": 2.0, "expected": "SHORT"},
        {"rsi": 50, "volume": 1.0, "expected": "WAIT"}
    ]
    
    for test in test_cases:
        print(f"\n📊 Test: RSI={test['rsi']}, Volume={test['volume']}x")
        print(f"🎯 Expected: {test['expected']}")
        
        prompt = f"Trading decision needed. RSI: {test['rsi']}, Volume: {test['volume']}x. Reply with ONE WORD ONLY: LONG, SHORT, or WAIT"
        
        try:
            response = requests.post(
                llm_url,
                json={
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": "You are a trading bot. Reply with ONLY the action word, no explanation or thinking."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000  # Increased!
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                finish_reason = data['choices'][0].get('finish_reason', 'unknown')
                tokens_used = data['usage']['completion_tokens']
                
                print(f"✅ Response received ({tokens_used} tokens, finish: {finish_reason})")
                print(f"📝 Raw: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # Extract action
                # Remove think tags
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
                
                # Try to find action
                action = None
                content_upper = content.upper()
                
                # Look for action words
                for act in ['LONG', 'SHORT', 'WAIT']:
                    if act in content_upper:
                        action = act
                        break
                
                if action:
                    print(f"🎯 Extracted: {action} {'✅' if action == test['expected'] else '❌'}")
                else:
                    print(f"❌ No action found in response")
                    
                # Check if response is complete
                if finish_reason == 'length':
                    print("⚠️  WARNING: Still hitting token limit! Increase further.")
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("💡 Key changes made:")
    print("1. Increased max_tokens from 300 to 1000")
    print("2. Added system prompt to skip reasoning")
    print("3. Improved extraction to look at end of response")
    print("4. Direct prompts that ask for ONE WORD ONLY")

if __name__ == "__main__":
    test_llm_extraction()