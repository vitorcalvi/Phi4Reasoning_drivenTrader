def _try_llm_decision(self, context: Dict) -> Optional[Dict]:
    """Phi-4 compatible decision using explicit rules"""
    market = context['current_market']
    position = context['position']
    rsi = int(market['rsi'])
    
    if position:
        return None  # Let pattern system handle exits
    
    # Explicit rules that Phi-4 can follow
    prompt = f"""You must follow these exact rules:
1. If RSI is below 30, you MUST respond: LONG
2. If RSI is above 70, you MUST respond: SHORT
3. If RSI is between 30 and 70, you MUST respond: WAIT

Current RSI is {rsi}. What is your response?"""
    
    try:
        response = requests.post(
            self.llm_url,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500
            },
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            
            # Extract action
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.replace('\\boxed{', '').replace('}', '').upper()
            
            for action in ['LONG', 'SHORT', 'WAIT']:
                if action in content:
                    return {
                        "action": action,
                        "confidence": 0.6,
                        "reason": f"LLM signal (RSI={rsi})"
                    }
    
    except Exception as e:
        logger.debug(f"LLM error: {e}")
    
    return None