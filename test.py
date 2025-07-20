#!/usr/bin/env python3
"""
Test script to verify LM Studio connection and model responses
"""
import requests
import json
import sys
from datetime import datetime


def test_lm_studio_connection(url="http://localhost:1234"):
    """Test basic connection to LM Studio"""
    print(f"🔍 Testing connection to LM Studio at {url}")
    
    try:
        # Test if server is running
        response = requests.get(url, timeout=5)
        print(f"✅ LM Studio server is running")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to LM Studio at {url}")
        print("📝 Make sure LM Studio is running and server is started")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_model_response(url="http://localhost:1234", model="local-model"):
    """Test if model can generate proper responses"""
    print(f"\n🤖 Testing model response...")
    
    endpoint = f"{url}/v1/chat/completions"
    
    # Simple test prompt
    test_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond in JSON format."
            },
            {
                "role": "user",
                "content": "Respond with a JSON object containing: status (string) and timestamp (string). Nothing else."
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(endpoint, json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"✅ Model responded successfully")
                print(f"📄 Response: {content}")
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(content)
                    print(f"✅ Response is valid JSON")
                    return True
                except json.JSONDecodeError:
                    print(f"⚠️ Response is not valid JSON - model may need better prompting")
                    return False
            else:
                print(f"❌ Unexpected response format")
                return False
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out - model may be too slow or not loaded")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_trading_decision(url="http://localhost:1234", model="local-model"):
    """Test if model can make trading decisions"""
    print(f"\n💹 Testing trading decision capability...")
    
    endpoint = f"{url}/v1/chat/completions"
    
    # Trading decision prompt
    trading_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """You are an expert crypto trading AI. Analyze market data and make trading decisions.
Always respond with valid JSON only. Consider risk management and market conditions carefully."""
            },
            {
                "role": "user",
                "content": """
TRADING DECISION REQUIRED

CURRENT MARKET DATA:
- Price: $2,345.67
- RSI: 32.5 (oversold)
- Volume Ratio: 2.3x average
- Spread: 0.03%
- Order Book Imbalance: 0.15 (bid heavy)

POSITION: No position

PERFORMANCE:
- Recent Win Rate: 60%
- Consecutive Losses: 0

Respond with JSON only:
{
    "action": "LONG" | "SHORT" | "WAIT",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation",
    "position_size": number (USDT),
    "stop_loss": price,
    "take_profit": price
}
"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        print("⏳ Waiting for model response (this may take a moment)...")
        response = requests.post(endpoint, json=trading_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    
                    try:
                        decision = json.loads(json_str)
                        
                        # Validate decision
                        required_fields = ['action', 'confidence', 'reason']
                        valid = all(field in decision for field in required_fields)
                        
                        if valid:
                            print(f"✅ Model made a valid trading decision!")
                            print(f"🎯 Decision: {decision['action']}")
                            print(f"🔮 Confidence: {decision['confidence']}")
                            print(f"💭 Reason: {decision['reason']}")
                            return True
                        else:
                            print(f"⚠️ Decision missing required fields")
                            print(f"📄 Decision: {json.dumps(decision, indent=2)}")
                            return False
                            
                    except json.JSONDecodeError:
                        print(f"❌ Failed to parse trading decision as JSON")
                        print(f"📄 Raw response: {content[:500]}...")
                        return False
                else:
                    print(f"❌ No JSON found in response")
                    print(f"📄 Raw response: {content[:500]}...")
                    return False
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 LM Studio Integration Test for Trading Bot")
    print("=" * 50)
    
    # Get URL from command line or use default
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"
    model = sys.argv[2] if len(sys.argv) > 2 else "local-model"
    
    print(f"📍 URL: {url}")
    print(f"🤖 Model: {model}")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_lm_studio_connection(url):
        tests_passed += 1
    else:
        print("\n❌ Cannot proceed without LM Studio connection")
        print("📝 Please:")
        print("   1. Start LM Studio")
        print("   2. Load a model")
        print("   3. Start the server")
        print("   4. Run this test again")
        return
    
    if test_model_response(url, model):
        tests_passed += 1
    
    if test_trading_decision(url, model):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your LM Studio setup is ready for trading.")
        print("\n🎯 Next steps:")
        print("   1. Configure your .env file")
        print("   2. Run: python main.py")
    else:
        print("⚠️ Some tests failed. Please check:")
        print("   - Model is properly loaded in LM Studio")
        print("   - Model supports instruction following")
        print("   - Server settings are correct")
        print("\n💡 Tip: Try a different model if current one fails JSON formatting")


if __name__ == "__main__":
    main()