#!/usr/bin/env python3
"""
AI-Driven Trading Bot - Main Entry Point
"""
import asyncio
import signal
import sys
import logging
from datetime import datetime
from config import Config
from core.bot_engine import BotEngine
from core.llm_engine import LLMEngine
from risk.risk_management import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        
        # Initialize components
        self.bot_engine = BotEngine(self.config)
        self.llm_engine = LLMEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown"""
        logger.info("🛑 Shutdown signal received")
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing AI-Driven Trading Bot")
        
        # Test connections
        if not await self.bot_engine.test_connection():
            logger.error("Failed to connect to exchange")
            return False
        
        if not await self.llm_engine.test_connection():
            logger.error("Failed to connect to LLM")
            return False
        
        # Load strategies
        strategies = await self.llm_engine.load_strategies()
        logger.info(f"📚 Loaded {len(strategies)} strategies")
        
        # Initialize risk manager
        self.risk_manager.initialize()
        
        logger.info("✅ All systems ready")
        return True
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("🔄 Starting AI trading loop")
        
        while self.running:
            try:
                # 1. Get market data
                market_data = await self.bot_engine.get_market_data()
                if not market_data:
                    await asyncio.sleep(5)
                    continue
                
                # 2. Get current position
                position = await self.bot_engine.get_current_position()
                
                # 3. Get performance
                performance = await self.bot_engine.get_performance_metrics()
                
                # 4. AI decision
                decision = await self.llm_engine.make_decision(
                    market_data=market_data,
                    position=position,
                    performance=performance
                )
                
                # 5. Risk validation
                validated_decision = await self.risk_manager.validate_decision(
                    decision=decision,
                    market_data=market_data,
                    position=position,
                    performance=performance
                )
                
                # 6. Execute
                if validated_decision['action'] != 'WAIT':
                    await self.bot_engine.execute_decision(validated_decision)
                
                # 7. Status
                await self._log_status(market_data, position, validated_decision)
                
                # 8. Sleep
                await asyncio.sleep(self.config.LOOP_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                await asyncio.sleep(10)
    
    async def _log_status(self, market_data, position, decision):
        """Log status"""
        price = market_data.get('price', 0)
        status = f"💹 {self.config.SYMBOL}: ${price:.4f}"
        
        if position:
            pnl = position.get('unrealized_pnl', 0)
            pnl_pct = position.get('pnl_percent', 0)
            status += f" | {position['side']} | PnL: ${pnl:.2f} ({pnl_pct:.1f}%)"
        else:
            status += " | No position"
        
        status += f" | {decision['action']}"
        
        if decision.get('confidence'):
            status += f" ({decision['confidence']:.0%})"
        
        logger.info(status)
    
    async def cleanup(self):
        """Cleanup on shutdown"""
        logger.info("🧹 Cleaning up...")
        
        # Close position
        position = await self.bot_engine.get_current_position()
        if position:
            logger.info("📊 Closing position")
            await self.bot_engine.close_position("Bot Shutdown")
        
        # Save data
        await self.bot_engine.save_performance_data()
        
        logger.info("✅ Cleanup complete")


async def main():
    """Main entry"""
    bot = TradingBot()
    
    # Initialize
    if not await bot.initialize():
        logger.error("Failed to initialize")
        return 1
    
    try:
        # Run
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    finally:
        # Cleanup
        await bot.cleanup()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)