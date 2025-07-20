"""
Optional monitoring API for the trading bot
Provides REST endpoints for status, control, and performance monitoring
"""
import asyncio
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import logging

logger = logging.getLogger(__name__)


class MonitoringAPI:
    def __init__(self, bot_engine, llm_engine, risk_manager, config):
        self.bot_engine = bot_engine
        self.llm_engine = llm_engine
        self.risk_manager = risk_manager
        self.config = config
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.commands_queue = []
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get current bot status"""
            try:
                position = asyncio.run(self.bot_engine.get_current_position())
                performance = asyncio.run(self.bot_engine.get_performance_metrics())
                risk_status = self.risk_manager.get_risk_status()
                decision_stats = self.llm_engine.get_decision_stats()
                
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': self.config.SYMBOL,
                    'position': position,
                    'performance': performance,
                    'risk': risk_status,
                    'decisions': decision_stats,
                    'trading_allowed': risk_status.get('trading_allowed', True)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/market', methods=['GET'])
        def get_market():
            """Get current market data"""
            try:
                market_data = asyncio.run(self.bot_engine.get_market_data())
                return jsonify(market_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/trades', methods=['GET'])
        def get_trades():
            """Get trade history"""
            try:
                limit = request.args.get('limit', 50, type=int)
                trades = self.bot_engine.trade_history[-limit:]
                
                return jsonify({
                    'trades': trades,
                    'total': len(self.bot_engine.trade_history),
                    'showing': len(trades)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/command', methods=['POST'])
        def send_command():
            """Send command to bot"""
            try:
                data = request.json
                command = data.get('command', '').upper()
                params = data.get('params', {})
                
                valid_commands = [
                    'FORCE_LONG', 'FORCE_SHORT', 'CLOSE_POSITION',
                    'PAUSE_TRADING', 'RESUME_TRADING', 'EMERGENCY_STOP',
                    'UPDATE_STRATEGY', 'UPDATE_RISK_PARAMS'
                ]
                
                if command not in valid_commands:
                    return jsonify({
                        'error': f'Invalid command. Valid commands: {valid_commands}'
                    }), 400
                
                self.commands_queue.append({
                    'command': command,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                })
                
                return jsonify({
                    'status': 'queued',
                    'command': command,
                    'queue_length': len(self.commands_queue)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/strategies', methods=['GET'])
        def get_strategies():
            """Get available strategies"""
            try:
                return jsonify({
                    'available': list(self.llm_engine.strategies.keys()),
                    'current': self.llm_engine.current_strategy,
                    'strategies': self.llm_engine.strategies
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Get current configuration (sanitized)"""
            try:
                # Don't expose sensitive data
                safe_config = {
                    'symbol': self.config.SYMBOL,
                    'timeframe': self.config.TIMEFRAME,
                    'testnet': self.config.TESTNET,
                    'position_size': self.config.DEFAULT_POSITION_SIZE,
                    'max_position_size': self.config.MAX_POSITION_SIZE,
                    'max_daily_loss': self.config.MAX_DAILY_LOSS,
                    'max_drawdown': self.config.MAX_DRAWDOWN,
                    'llm_model': self.config.LLM_MODEL
                }
                return jsonify(safe_config)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/performance/chart', methods=['GET'])
        def get_performance_chart():
            """Get data for performance chart"""
            try:
                trades = self.bot_engine.trade_history
                
                if not trades:
                    return jsonify({'data': [], 'summary': {}})
                
                # Calculate cumulative P&L
                cumulative_pnl = []
                total = 0
                
                for trade in trades:
                    total += trade['pnl']
                    cumulative_pnl.append({
                        'timestamp': trade['exit_time'],
                        'pnl': trade['pnl'],
                        'cumulative_pnl': total,
                        'side': trade['side']
                    })
                
                # Summary statistics
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] < 0]
                
                summary = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(trades) if trades else 0,
                    'total_pnl': total,
                    'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                    'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                    'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0
                }
                
                return jsonify({
                    'data': cumulative_pnl,
                    'summary': summary
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            })
    
    def process_commands(self):
        """Process queued commands"""
        while self.commands_queue:
            cmd = self.commands_queue.pop(0)
            command = cmd['command']
            params = cmd['params']
            
            try:
                if command == 'EMERGENCY_STOP':
                    self.risk_manager.emergency_stop()
                    logger.warning("🚨 Emergency stop activated via API")
                
                elif command == 'UPDATE_STRATEGY':
                    strategy = params.get('strategy')
                    if strategy:
                        self.llm_engine.update_strategy(strategy)
                        logger.info(f"📝 Strategy updated to: {strategy}")
                
                # Add more command handlers as needed
                
            except Exception as e:
                logger.error(f"❌ Error processing command {command}: {e}")
    
    def start(self):
        """Start the monitoring API server"""
        def run_server():
            self.app.run(
                host=self.config.API_HOST,
                port=self.config.API_PORT,
                debug=False
            )
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"🌐 Monitoring API started on http://{self.config.API_HOST}:{self.config.API_PORT}")
        logger.info("📊 Endpoints:")
        logger.info("  GET  /status          - Current bot status")
        logger.info("  GET  /market          - Market data")
        logger.info("  GET  /trades          - Trade history")
        logger.info("  GET  /strategies      - Available strategies")
        logger.info("  GET  /performance/chart - Performance chart data")
        logger.info("  POST /command         - Send commands to bot")


# Example usage in main.py:
"""
# Add to main.py after initializing components:

from api.monitoring import MonitoringAPI

# In TradingBot.__init__:
self.monitoring_api = MonitoringAPI(
    self.bot_engine,
    self.llm_engine,
    self.risk_manager,
    self.config
)

# In TradingBot.initialize:
self.monitoring_api.start()

# In main loop, process commands:
self.monitoring_api.process_commands()
"""