#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口點
這個文件結合了配置管理和自動化交易執行功能

功能:
1. 環境配置和日誌管理
2. 模型加載和管理
3. 自動化交易執行
4. 數據監控和錯誤處理

使用方法:
1. 設置環境變量（參考 .env.example）
2. 配置股票列表（stock_list.csv）
3. 運行程序：
   python main.py

示例 .env 文件:
API_KEY=your_api_key_here
MODEL_PATH=models/
DEBUG=False
"""

import os
import time
import schedule
from pathlib import Path
from typing import Dict, Any, List
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import joblib

# 加載環境變量
load_dotenv()

# 配置日誌
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

def load_config() -> Dict[str, Any]:
    """
    從環境變量加載配置
    
    返回:
        Dict[str, Any]: 配置字典
    
    示例:
        config = load_config()
        api_key = config['api_key']
    """
    return {
        "api_key": os.getenv("API_KEY"),
        "model_path": os.getenv("MODEL_PATH", "models/"),
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "schedule_time": os.getenv("SCHEDULE_TIME", "09:30")
    }

def load_models(models_folder: str) -> Dict[str, Any]:
    """
    加載訓練好的模型
    
    參數:
        models_folder: 模型文件夾路徑
    
    返回:
        Dict[str, Any]: 模型字典
    
    示例:
        models = load_models("models/")
        prediction = models['stock_predictor'].predict(data)
    """
    models = {}
    try:
        model_files = Path(models_folder).glob("*.pkl")
        for model_file in model_files:
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
            logger.info(f"加載模型: {model_name}")
    except Exception as e:
        logger.error(f"模型加載失敗: {str(e)}")
    return models

def load_stock_list(stock_list_file: str) -> List[str]:
    """
    讀取股票列表
    
    參數:
        stock_list_file: 股票列表文件路徑
    
    返回:
        List[str]: 股票代碼列表
    
    示例:
        stocks = load_stock_list("data/stock_list.csv")
        for stock in stocks:
            print(f"處理股票: {stock}")
    """
    try:
        stock_list = pd.read_csv(stock_list_file)
        return stock_list['Symbol'].tolist()
    except Exception as e:
        logger.error(f"讀取股票列表失敗: {str(e)}")
        return []

def get_latest_data(stock_list: List[str]) -> pd.DataFrame:
    """
    獲取股票列表中每隻股票的最新數據
    
    參數:
        stock_list: 股票代碼列表
    
    返回:
        pd.DataFrame: 最新的股票數據
    
    示例:
        data = get_latest_data(['AAPL', 'GOOGL'])
        print(data.head())
    """
    all_data = []
    for symbol in stock_list:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2d")
            if not hist.empty:
                hist['Symbol'] = symbol
                all_data.append(hist)
        except Exception as e:
            logger.error(f"獲取{symbol}數據失敗: {str(e)}")
    
    return pd.concat(all_data) if all_data else pd.DataFrame()

def make_predictions(models: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
    """
    使用加載的模型對最新數據進行預測
    
    參數:
        models: 模型字典
        data: 輸入數據
    
    返回:
        Dict[str, Any]: 預測結果
    
    示例:
        predictions = make_predictions(models, market_data)
        for symbol, pred in predictions.items():
            print(f"{symbol}: {pred}")
    """
    predictions = {}
    try:
        for symbol, group in data.groupby('Symbol'):
            if 'stock_predictor' in models:
                pred = models['stock_predictor'].predict(group)
                predictions[symbol] = pred
    except Exception as e:
        logger.error(f"預測失敗: {str(e)}")
    return predictions

def execute_trades(predictions: Dict[str, Any]):
    """
    根據預測結果執行買賣操作
    
    參數:
        predictions: 預測結果字典
    
    示例:
        execute_trades({'AAPL': 1, 'GOOGL': -1})
    """
    try:
        for symbol, prediction in predictions.items():
            if prediction > 0:
                logger.info(f"買入信號: {symbol}")
                # 實現你的買入邏輯
            elif prediction < 0:
                logger.info(f"賣出信號: {symbol}")
                # 實現你的賣出邏輯
    except Exception as e:
        logger.error(f"交易執行失敗: {str(e)}")

def scanner():
    """
    主要掃描功能
    執行完整的交易流程：加載模型 -> 獲取數據 -> 預測 -> 交易
    """
    try:
        config = load_config()
        models = load_models(config['model_path'])
        stock_list = load_stock_list("data/stock_list.csv")
        
        if not stock_list:
            logger.warning("股票列表為空")
            return
            
        data = get_latest_data(stock_list)
        if data.empty:
            logger.warning("未獲取到市場數據")
            return
            
        predictions = make_predictions(models, data)
        if predictions:
            execute_trades(predictions)
        
    except Exception as e:
        logger.error(f"掃描過程發生錯誤: {str(e)}")

def main():
    """
    主程序入口
    設置定時任務並保持程序運行
    """
    try:
        config = load_config()
        schedule_time = config.get('schedule_time', '09:30')
        
        # 設置定時任務
        schedule.every().day.at(schedule_time).do(scanner)
        logger.info(f"程序已啟動，將在每天 {schedule_time} 執行掃描")
        
        # 保持程序運行
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"程序運行錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 