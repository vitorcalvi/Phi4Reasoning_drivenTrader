def _try_llm_decision(self, context: Dict) -> Optional[Dict]:
    """Disabled LLM - using patterns only"""
    # LLM not working reliably with phi-4-mini-reasoning
    # Returning None to use pattern-based decisions
    return None