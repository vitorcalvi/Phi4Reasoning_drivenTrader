#!/usr/bin/env python3
"""
Debug script to identify JSON parsing issues with Phi-4
"""
import requests
import json
import sys
import time


def test_trading_decision_debug(url="http://localhost:1234", model="local-model"):
    """Test trading decisions with detailed debugging"""
    
    # Simulated market conditions
    test_cases = [
        {
            "name": "Oversold",
            "price": 3700.00,
            "rsi": 28.5,
            "volume": 2.3,
            "position": "LONG"
        },
        {
            "name": "Overbought",
            "price": 3750.00,
            "rsi": 72.0,
            "volume": 1.8,
            "position": "SHORT"
        },
        {
            "name": "Neutral",
            "price": 3720.00,
            "rsi": 50.0,
            "volume": 1.0,
            "position": None
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test['name']} Condition")
        print(f"{'='*60}")
        
        # Build prompt similar to the bot
        prompt = f"""Price: ${test['price']:.2f}
RSI: {test['rsi']:.1f}
Volume: {test['volume']:.1f}x
Position: {test['position'] if test['position'] else 'None'}

Output only this JSON:
{{
    "action": "LONG",
    "confidence": 0.8,
    "reason": "oversold"
}}

action: LONG, SHORT, CLOSE, or WAIT"""
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert crypto trading AI. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            print(f"📤 Sending request...")
            response = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                print(f"\n📥 Raw Response ({len(content)} chars):")
                print("-" * 40)
                print(content)
                print("-" * 40)
                
                # Check for <think> tags
                if '<think>' in content:
                    print("\n🧠 Found <think> tags, extracting content after </think>")
                    if '</think>' in content:
                        content = content.split('</think>')[-1].strip()
                        print(f"📄 Content after removing think tags:")
                        print(content)
                    else:
                        print("⚠️ WARNING: <think> tag found but no closing </think>!")
                
                # Try to find JSON
                json_start = content.find('{')
                if json_start != -1:
                    print(f"\n🔍 Found JSON starting at position {json_start}")
                    
                    # Show what's around position 20 (where error occurs)
                    if len(content) > 20:
                        print(f"📍 Content around char 20: '{content[15:25]}'")
                        print(f"📍 First 30 chars: '{content[:30]}'")
                    
                    # Find matching closing brace
                    brace_count = 0
                    json_end = json_start
                    for idx in range(json_start, len(content)):
                        if content[idx] == '{':
                            brace_count += 1
                        elif content[idx] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = idx + 1
                                break
                    
                    json_str = content[json_start:json_end]
                    print(f"\n📋 Extracted JSON string:")
                    print(json_str)
                    
                    # Show each character with position for debugging
                    print(f"\n🔬 Character-by-character analysis:")
                    for idx, char in enumerate(json_str[:40]):  # First 40 chars
                        if char == '\n':
                            print(f"  [{idx:2d}]: \\n (newline)")
                        elif char == '\t':
                            print(f"  [{idx:2d}]: \\t (tab)")
                        elif char == ' ':
                            print(f"  [{idx:2d}]: ' ' (space)")
                        else:
                            print(f"  [{idx:2d}]: '{char}'")
                    
                    try:
                        decision = json.loads(json_str)
                        print(f"\n✅ Successfully parsed JSON: {decision}")
                        
                        # Check for field mappings
                        if 'Decision' in decision:
                            print(f"📝 Found 'Decision' field (needs mapping to 'action')")
                        if 'Reasoning' in decision:
                            print(f"📝 Found 'Reasoning' field (needs mapping to 'reason')")
                            
                    except json.JSONDecodeError as e:
                        print(f"\n❌ JSON Parse Error: {e}")
                        print(f"❌ Error position: line {e.lineno}, column {e.colno} (char {e.pos})")
                        
                        # Try to identify the issue
                        if e.pos and e.pos < len(json_str):
                            error_context = json_str[max(0, e.pos-10):e.pos+10]
                            print(f"❌ Context around error: '{error_context}'")
                            print(f"❌ Character at error position: '{json_str[e.pos]}' (ASCII: {ord(json_str[e.pos])})")
                        
                        # Check for common issues
                        if '\n' in json_str[:e.pos]:
                            print("⚠️ Possible issue: Newline in JSON key/value")
                        if json_str.count('"') % 2 != 0:
                            print("⚠️ Possible issue: Odd number of quotes")
                        if 'true' in json_str or 'false' in json_str:
                            print("⚠️ Note: JSON uses lowercase true/false")
                else:
                    print("\n❌ No JSON found in response")
                    
            else:
                print(f"❌ Request failed with status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay between tests
        time.sleep(2)


def main():
    """Run debug tests"""
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"
    model = sys.argv[2] if len(sys.argv) > 2 else "local-model"
    
    print("🔍 JSON Parsing Debug Test for Phi-4")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Model: {model}")
    print("=" * 60)
    
    test_trading_decision_debug(url, model)
    
    print("\n" + "=" * 60)
    print("💡 Debug Summary:")
    print("- Check if JSON has unexpected characters at position 20")
    print("- Look for newlines or special characters in keys/values")
    print("- Verify all strings are properly quoted")
    print("- Ensure no trailing commas or syntax issues")


if __name__ == "__main__":
    main()