"""
Configuration file for LLM-Driven Trading Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Exchange Configuration
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'
    API_KEY = os.getenv('API_KEY', '')
    SECRET_KEY = os.getenv('SECRET_KEY', '')
    
    # Trading Configuration
    SYMBOL = os.getenv('SYMBOL', 'ETHUSDT')
    TIMEFRAME = os.getenv('TIMEFRAME', '1')  # 1 minute
    DEFAULT_POSITION_SIZE = float(os.getenv('DEFAULT_POSITION_SIZE', '10'))  # USDT
    MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE', '5'))  # USDT
    
    # LLM Configuration
    LLM_URL = os.getenv('LLM_URL', 'http://localhost:1234')  # Local Phi-4 or API endpoint
    LLM_MODEL = os.getenv('LLM_MODEL', 'microsoft/phi-4')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '1000'))
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '100'))  # USDT
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50'))  # USDT
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '10'))  # Percentage
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
    MIN_RISK_REWARD_RATIO = float(os.getenv('MIN_RISK_REWARD_RATIO', '1.5'))
    
    # System Configuration
    LOOP_INTERVAL = int(os.getenv('LOOP_INTERVAL', '5'))  # Seconds
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Performance Tracking
    SAVE_TRADES = os.getenv('SAVE_TRADES', 'True').lower() == 'true'
    TRADES_FILE = os.getenv('TRADES_FILE', 'trades.json')
    
    # API Server (for monitoring)
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    
    # Strategy defaults (can be overridden by YAML)
    DEFAULT_RSI_PERIOD = int(os.getenv('DEFAULT_RSI_PERIOD', '14'))
    DEFAULT_RSI_OVERSOLD = float(os.getenv('DEFAULT_RSI_OVERSOLD', '30'))
    DEFAULT_RSI_OVERBOUGHT = float(os.getenv('DEFAULT_RSI_OVERBOUGHT', '70'))
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.API_KEY:
            errors.append("API_KEY is required")
        
        if not cls.SECRET_KEY:
            errors.append("SECRET_KEY is required")
        
        if cls.DEFAULT_POSITION_SIZE < cls.MIN_POSITION_SIZE:
            errors.append("DEFAULT_POSITION_SIZE must be >= MIN_POSITION_SIZE")
        
        if cls.MAX_POSITION_SIZE < cls.DEFAULT_POSITION_SIZE:
            errors.append("MAX_POSITION_SIZE must be >= DEFAULT_POSITION_SIZE")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def display(cls):
        """Display configuration (hide sensitive data)"""
        print("🔧 Configuration:")
        print(f"  Exchange: {'Testnet' if cls.TESTNET else 'Mainnet'}")
        print(f"  Symbol: {cls.SYMBOL}")
        print(f"  Timeframe: {cls.TIMEFRAME}m")
        print(f"  Position Size: ${cls.DEFAULT_POSITION_SIZE} - ${cls.MAX_POSITION_SIZE}")
        print(f"  Max Daily Loss: ${cls.MAX_DAILY_LOSS}")
        print(f"  Max Drawdown: {cls.MAX_DRAWDOWN}%")
        print(f"  LLM: {cls.LLM_MODEL} @ {cls.LLM_URL}")
        print(f"  API Key: {'✓' if cls.API_KEY else '✗'}")
        print(f"  Secret Key: {'✓' if cls.SECRET_KEY else '✗'}")


# Create .env.example file content
ENV_EXAMPLE = """# Exchange Configuration
TESTNET=True
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# Trading Configuration
SYMBOL=ETHUSDT
TIMEFRAME=1
DEFAULT_POSITION_SIZE=10
MIN_POSITION_SIZE=5

# LLM Configuration
LLM_URL=http://localhost:1234
LLM_MODEL=microsoft/phi-4
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000

# Risk Management
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=50
MAX_DRAWDOWN=10
MAX_CONSECUTIVE_LOSSES=3
MIN_RISK_REWARD_RATIO=1.5

# System Configuration
LOOP_INTERVAL=5
LOG_LEVEL=INFO

# Performance Tracking
SAVE_TRADES=True
TRADES_FILE=trades.json

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Strategy Defaults
DEFAULT_RSI_PERIOD=14
DEFAULT_RSI_OVERSOLD=30
DEFAULT_RSI_OVERBOUGHT=70
"""

if __name__ == "__main__":
    # Validate and display configuration
    Config.validate()
    Config.display()
    
    # Create .env.example if it doesn't exist
    if not os.path.exists('.env.example'):
        with open('.env.example', 'w') as f:
            f.write(ENV_EXAMPLE)
        print("\n✅ Created .env.example file")