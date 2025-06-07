"""
數據服務模組
處理數據獲取、預處理和特徵工程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yfinance as yf
from loguru import logger
from sklearn.preprocessing import StandardScaler
import ta
from config.config import DATA_CONFIG

class DataService:
    """數據服務類"""
    
    def __init__(self, config: Dict = None):
        """
        初始化數據服務
        
        參數:
            config: 數據配置，如果為None則使用默認配置
        """
        self.config = config or DATA_CONFIG
        self.scaler = StandardScaler()
        
    def load_stock_list(self) -> List[str]:
        """
        加載股票列表
        
        返回:
            List[str]: 股票代碼列表
        """
        try:
            df = pd.read_csv(self.config['stock_list_file'])
            return df['Symbol'].tolist()
        except Exception as e:
            logger.error(f"加載股票列表失敗: {str(e)}")
            return []
            
    def get_stock_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        獲取股票歷史數據
        
        參數:
            symbol: 股票代碼
            period: 數據周期
            
        返回:
            Optional[pd.DataFrame]: 股票數據
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
            return None
        except Exception as e:
            logger.error(f"獲取股票數據失敗 {symbol}: {str(e)}")
            return None
            
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技術指標
        
        參數:
            df: 原始數據
            
        返回:
            pd.DataFrame: 添加技術指標後的數據
        """
        try:
            # 移動平均線
            df['MA5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['MA10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            return df
            
        except Exception as e:
            logger.error(f"添加技術指標失敗: {str(e)}")
            return df
            
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        準備特徵數據
        
        參數:
            df: 原始數據
            
        返回:
            Tuple[np.ndarray, List[str]]: 特徵矩陣和特徵名稱列表
        """
        try:
            # 選擇特徵列
            features = df[self.config['feature_columns']].copy()
            
            # 處理缺失值
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # 標準化
            features_scaled = self.scaler.fit_transform(features)
            
            return features_scaled, self.config['feature_columns']
            
        except Exception as e:
            logger.error(f"準備特徵失敗: {str(e)}")
            return np.array([]), []
            
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        保存數據到文件
        
        參數:
            df: 要保存的數據
            filename: 文件名
        """
        try:
            save_path = Path(self.config['data_dir']) / filename
            df.to_csv(save_path)
            logger.info(f"數據已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存數據失敗: {str(e)}")
            
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        從文件加載數據
        
        參數:
            filename: 文件名
            
        返回:
            Optional[pd.DataFrame]: 加載的數據
        """
        try:
            load_path = Path(self.config['data_dir']) / filename
            df = pd.read_csv(load_path)
            return df
        except Exception as e:
            logger.error(f"加載數據失敗: {str(e)}")
            return None
            
    def prepare_training_data(self, symbols: List[str], period: str = "2y") -> Tuple[np.ndarray, np.ndarray]:
        """
        準備訓練數據
        
        參數:
            symbols: 股票代碼列表
            period: 數據周期
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 特徵矩陣和標籤
        """
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            try:
                # 獲取數據
                df = self.get_stock_data(symbol, period)
                if df is None:
                    continue
                    
                # 添加技術指標
                df = self.add_technical_indicators(df)
                
                # 準備特徵
                features, _ = self.prepare_features(df)
                
                # 準備標籤（這裡使用簡單的漲跌作為標籤）
                labels = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[:-1]
                
                # 去掉最後一行（因為沒有下一天的標籤）
                features = features[:-1]
                
                all_features.append(features)
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"準備訓練數據失敗 {symbol}: {str(e)}")
                continue
                
        if not all_features:
            return np.array([]), np.array([])
            
        return np.vstack(all_features), np.concatenate(all_labels) 