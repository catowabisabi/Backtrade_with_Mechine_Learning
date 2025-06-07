"""
交易服務模組
處理所有交易相關的操作
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger
from config.config import TRADING_CONFIG

class TradingService:
    """交易服務類"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化交易服務
        
        參數:
            config: 交易配置，如果為None則使用默認配置
        """
        self.config = config or TRADING_CONFIG
        self.positions: Dict[str, Dict[str, Any]] = {}  # 當前持倉
        self.orders: List[Dict[str, Any]] = []          # 訂單歷史
        
    def check_position_limits(self, symbol: str) -> bool:
        """
        檢查是否可以開新倉位
        
        參數:
            symbol: 股票代碼
            
        返回:
            bool: 是否可以開新倉位
        """
        # 檢查總倉位數量
        if len(self.positions) >= self.config['max_positions']:
            logger.warning(f"達到最大倉位數量限制: {self.config['max_positions']}")
            return False
            
        # 檢查是否已有該股票的倉位
        if symbol in self.positions:
            logger.warning(f"已有 {symbol} 的倉位")
            return False
            
        return True
        
    def calculate_position_size(self, capital: float, price: float) -> int:
        """
        計算倉位大小
        
        參數:
            capital: 可用資金
            price: 當前價格
            
        返回:
            int: 可買入的股數
        """
        position_capital = capital * self.config['position_size']
        return int(position_capital / price)
        
    def execute_buy(self, symbol: str, price: float, size: int) -> bool:
        """
        執行買入操作
        
        參數:
            symbol: 股票代碼
            price: 買入價格
            size: 買入數量
            
        返回:
            bool: 是否執行成功
        """
        try:
            if not self.check_position_limits(symbol):
                return False
                
            position = {
                'symbol': symbol,
                'entry_price': price,
                'size': size,
                'stop_loss': price * (1 - self.config['stop_loss']),
                'take_profit': price * (1 + self.config['take_profit'])
            }
            
            self.positions[symbol] = position
            self.orders.append({
                'symbol': symbol,
                'type': 'BUY',
                'price': price,
                'size': size,
                'timestamp': pd.Timestamp.now()
            })
            
            logger.info(f"買入 {symbol}: 價格={price}, 數量={size}")
            return True
            
        except Exception as e:
            logger.error(f"買入執行失敗 {symbol}: {str(e)}")
            return False
            
    def execute_sell(self, symbol: str, price: float) -> bool:
        """
        執行賣出操作
        
        參數:
            symbol: 股票代碼
            price: 賣出價格
            
        返回:
            bool: 是否執行成功
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"沒有 {symbol} 的倉位")
                return False
                
            position = self.positions[symbol]
            self.orders.append({
                'symbol': symbol,
                'type': 'SELL',
                'price': price,
                'size': position['size'],
                'timestamp': pd.Timestamp.now()
            })
            
            profit = (price - position['entry_price']) * position['size']
            logger.info(f"賣出 {symbol}: 價格={price}, 數量={position['size']}, 盈虧={profit:.2f}")
            
            del self.positions[symbol]
            return True
            
        except Exception as e:
            logger.error(f"賣出執行失敗 {symbol}: {str(e)}")
            return False
            
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        檢查止損止盈
        
        參數:
            symbol: 股票代碼
            current_price: 當前價格
            
        返回:
            Optional[str]: 如果需要執行操作，返回 'SELL'，否則返回 None
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        if current_price <= position['stop_loss']:
            logger.info(f"{symbol} 觸發止損: 價格={current_price}")
            return 'SELL'
            
        if current_price >= position['take_profit']:
            logger.info(f"{symbol} 觸發止盈: 價格={current_price}")
            return 'SELL'
            
        return None
        
    def get_position_summary(self) -> pd.DataFrame:
        """
        獲取倉位摘要
        
        返回:
            pd.DataFrame: 倉位摘要
        """
        if not self.positions:
            return pd.DataFrame()
            
        return pd.DataFrame(self.positions.values())
        
    def get_order_history(self) -> pd.DataFrame:
        """
        獲取訂單歷史
        
        返回:
            pd.DataFrame: 訂單歷史
        """
        if not self.orders:
            return pd.DataFrame()
            
        return pd.DataFrame(self.orders) 