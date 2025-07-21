def _try_llm_decision(self, context: Dict) -> Optional[Dict]:
    """Optimized LLM decision extraction"""
    market = context['current_market']
    position = context['position']
    
    # Prompt that works best with this model
    if position:
        prompt = f"Current position PNL: {position.get('pnl_percent', 0):.1f}%. RSI: {int(market['rsi'])}. Should I CLOSE or WAIT?"
    else:
        prompt = f"RSI: {int(market['rsi'])}, Volume: {market['volume_ratio']:.1f}x. Should I go LONG, SHORT, or WAIT?"
    
    try:
        response = requests.post(
            self.llm_url,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300
            },
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            logger.debug(f"LLM response: {content[:100]}...")
            
            # Remove think tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            
            # Find most common action word
            content_upper = content.upper()
            counts = {}
            for act in ['LONG', 'SHORT', 'WAIT', 'CLOSE']:
                pattern = r'\b' + act + r'\b'
                counts[act] = len(re.findall(pattern, content_upper))
            
            # Get action with most mentions
            if any(counts.values()):
                action = max(counts.items(), key=lambda x: x[1])[0]
            else:
                action = None
            
            # Validate and return decision
            if action:
                if action == 'LONG' and not position:
                    return {"action": "LONG", "confidence": 0.6, "reason": "LLM signal"}
                elif action == 'SHORT' and not position:
                    return {"action": "SHORT", "confidence": 0.6, "reason": "LLM signal"}
                elif action == 'CLOSE' and position:
                    return {"action": "CLOSE", "confidence": 0.7, "reason": "LLM signal"}
                elif action == 'WAIT':
                    return {"action": "WAIT", "confidence": 0.5, "reason": "LLM signal"}
            
            logger.debug("No valid action extracted from LLM")
            
    except Exception as e:
        logger.debug(f"LLM error: {str(e)[:50]}")
    
    return None