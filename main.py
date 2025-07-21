#!/usr/bin/env python3
"""
True AI-Driven Autonomous Trading Bot
"""
import asyncio
import signal
import sys
import logging
from datetime import datetime
from typing import Dict, Optional
from config import Config
from core.bot_engine import BotEngine
from core.llm_engine import LLMEngine
from risk.risk_management import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AITradingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        
        # Initialize components
        self.bot_engine = BotEngine(self.config)
        self.llm_engine = LLMEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Connect bot and LLM for learning feedback
        self.bot_engine.set_llm_engine(self.llm_engine)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance tracking
        self.loop_count = 0
        self.start_time = datetime.now()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown"""
        logger.info("🛑 Shutdown signal received")
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing True AI-Driven Autonomous Trading Bot")
        
        # Test connections
        if not await self.bot_engine.test_connection():
            logger.error("Failed to connect to exchange")
            return False
        
        if not await self.llm_engine.test_connection():
            logger.info("⚠️  LLM offline - using pattern-based decisions")
        
        # Load AI patterns
        patterns = await self.llm_engine.load_strategies()
        logger.info(f"🧠 AI Ready with {patterns.get('patterns', 0)} learned patterns")
        
        # Initialize risk manager
        self.risk_manager.initialize()
        
        logger.info("✅ AI Systems operational")
        return True
    
    async def check_position_exits(self, position: Dict, market_data: Dict):
        """Check if position should be closed based on stops/targets"""
        if not position:
            return
        
        current_price = market_data.get('price', 0)
        
        # Check stop loss
        if position['side'] == 'LONG':
            if current_price <= position['stop_loss']:
                await self.bot_engine.close_position(f"Stop loss hit @ ${current_price:.4f}")
                return
        else:  # SHORT
            if current_price >= position['stop_loss']:
                await self.bot_engine.close_position(f"Stop loss hit @ ${current_price:.4f}")
                return
        
        # Check take profit
        if position['side'] == 'LONG':
            if current_price >= position['take_profit']:
                await self.bot_engine.close_position(f"Take profit hit @ ${current_price:.4f}")
                return
        else:  # SHORT
            if current_price <= position['take_profit']:
                await self.bot_engine.close_position(f"Take profit hit @ ${current_price:.4f}")
                return
    
    async def run(self):
        """Main AI trading loop"""
        self.running = True
        logger.info("🔄 Starting AI autonomous trading")
        
        while self.running:
            try:
                self.loop_count += 1
                
                # 1. Get market data
                market_data = await self.bot_engine.get_market_data()
                if not market_data:
                    await asyncio.sleep(5)
                    continue
                
                # 2. Get current position
                position = await self.bot_engine.get_current_position()
                
                # 3. Check position exits first (stop loss/take profit)
                await self.check_position_exits(position, market_data)
                
                # Re-check position after potential exit
                position = await self.bot_engine.get_current_position()
                
                # 4. Get performance
                performance = await self.bot_engine.get_performance_metrics()
                
                # 5. AI autonomous decision
                decision = await self.llm_engine.make_decision(
                    market_data=market_data,
                    position=position,
                    performance=performance
                )
                
                # 6. Risk validation
                validated_decision = await self.risk_manager.validate_decision(
                    decision=decision,
                    market_data=market_data,
                    position=position,
                    performance=performance
                )
                
                # 7. Execute if not waiting
                if validated_decision['action'] != 'WAIT':
                    await self.bot_engine.execute_decision(validated_decision)
                
                # 8. Log AI status
                await self._log_ai_status(market_data, position, validated_decision)
                
                # 9. Periodic AI learning save
                if self.loop_count % 20 == 0:  # Save more frequently
                    self.llm_engine.save_patterns()
                    logger.info(f"💾 Patterns saved (loop {self.loop_count})")
                
                # 10. Sleep
                await asyncio.sleep(self.config.LOOP_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(10)
    
    async def _log_ai_status(self, market_data, position, decision):
        """Log AI status with learning info"""
        price = market_data.get('price', 0)
        indicators = market_data.get('indicators', {})
        
        status = f"#{self.loop_count} ${price:.4f} | RSI: {indicators.get('rsi', 0):.0f} | Vol: {indicators.get('volume_ratio', 1):.1f}x"
        
        if position:
            pnl = position.get('unrealized_pnl', 0)
            pnl_pct = position.get('pnl_percent', 0)
            duration = position.get('duration_seconds', 0) / 60
            status += f" | {position['side']} PnL: ${pnl:.2f} ({pnl_pct:.1f}%) {duration:.0f}m"
        
        status += f" | {decision['action']}"
        
        if decision.get('confidence'):
            status += f" ({decision['confidence']:.0%})"
        
        # Add AI learning status
        stats = self.llm_engine.get_decision_stats()
        patterns = stats.get('patterns_learned', 0)
        if patterns > 0:
            status += f" | 🧠 {patterns}"
        
        logger.info(status)
        
        # Log detailed info every 10 loops
        if self.loop_count % 10 == 0 and decision.get('reason'):
            logger.info(f"   → {decision.get('reason')}")
    
    async def cleanup(self):
        """Cleanup on shutdown"""
        logger.info("🧹 Cleaning up...")
        
        # Close position
        position = await self.bot_engine.get_current_position()
        if position:
            logger.info("📊 Closing position before shutdown")
            await self.bot_engine.close_position("Bot Shutdown")
        
        # Save all data
        await self.bot_engine.save_performance_data()
        
        # Final AI stats
        stats = self.llm_engine.get_decision_stats()
        runtime = (datetime.now() - self.start_time).total_seconds() / 60
        
        logger.info("=" * 60)
        logger.info("📊 AI Performance Summary:")
        logger.info(f"   Runtime: {runtime:.1f} minutes")
        logger.info(f"   Total loops: {self.loop_count}")
        logger.info(f"   Patterns learned: {stats.get('patterns_learned', 0)}")
        logger.info(f"   Trades completed: {len(self.bot_engine.trade_history)}")
        
        if self.bot_engine.trade_history:
            wins = len([t for t in self.bot_engine.trade_history if t['pnl'] > 0])
            total = len(self.bot_engine.trade_history)
            logger.info(f"   Win rate: {wins}/{total} ({wins/total*100:.0f}%)")
            logger.info(f"   Total P&L: ${self.bot_engine.performance_data['total_pnl']:.2f}")
        
        if stats.get('action_distribution'):
            logger.info(f"   Decisions: {stats['action_distribution']}")
        
        logger.info("=" * 60)
        logger.info("✅ Cleanup complete")


async def main():
    """Main entry"""
    bot = AITradingBot()
    
    # Initialize
    if not await bot.initialize():
        logger.error("Failed to initialize AI systems")
        return 1
    
    try:
        # Run autonomous AI
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Cleanup
        await bot.cleanup()
    
    return 0


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🤖 True AI Autonomous Trading Bot")
    logger.info("🧠 Pattern-based learning with LLM assist")
    logger.info("📈 Learns from every trade outcome")
    logger.info("=" * 60)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)