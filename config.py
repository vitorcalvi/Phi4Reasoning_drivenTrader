"""
Configuration for AI-Driven Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Exchange
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'
    API_KEY = os.getenv('API_KEY', '')
    SECRET_KEY = os.getenv('SECRET_KEY', '')
    
    # Trading
    SYMBOL = os.getenv('SYMBOL', 'ETHUSDT')
    TIMEFRAME = os.getenv('TIMEFRAME', '1')
    DEFAULT_POSITION_SIZE = float(os.getenv('DEFAULT_POSITION_SIZE', '10'))
    MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE', '5'))
    
    # LLM
    LLM_URL = os.getenv('LLM_URL', 'http://localhost:1234')
    LLM_MODEL = os.getenv('LLM_MODEL', 'local-model')
    SIMPLE_PROMPTS = os.getenv('SIMPLE_PROMPTS', 'True').lower() == 'true'
    
    # Risk
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '100'))
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50'))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
    MIN_RISK_REWARD_RATIO = float(os.getenv('MIN_RISK_REWARD_RATIO', '1.5'))
    
    # System
    LOOP_INTERVAL = int(os.getenv('LOOP_INTERVAL', '5'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        """Validate config"""
        if not cls.API_KEY or not cls.SECRET_KEY:
            raise ValueError("API_KEY and SECRET_KEY required")
        return True
    
    @classmethod
    def display(cls):
        """Display config"""
        print("🔧 Configuration:")
        print(f"  Exchange: {'Testnet' if cls.TESTNET else 'Mainnet'}")
        print(f"  Symbol: {cls.SYMBOL}")
        print(f"  Timeframe: {cls.TIMEFRAME}m")
        print(f"  LLM: {cls.LLM_MODEL} @ {cls.LLM_URL}")


if __name__ == "__main__":
    Config.validate()
    Config.display()