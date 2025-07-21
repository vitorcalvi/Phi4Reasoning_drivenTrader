
def _extract_llm_action(self, content: str, has_position: bool) -> Optional[str]:
    """Extract action from LLM response"""
    
    # Fallback: look anywhere
    content_upper = content.upper()
    for action in ['LONG', 'SHORT', 'WAIT', 'CLOSE']:
        if action in content_upper:
            pattern = r'\b' + action + r'\b'
            if re.search(pattern, content_upper):
                return action
    return None
