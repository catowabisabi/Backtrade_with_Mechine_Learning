"""
主程序
自動交易系統的入口點
"""

import schedule
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Any
import sys

from services.data_service import DataService
from services.trading import TradingService
from config.config import get_all_config, LOGGING_CONFIG, LOG_DIR

def setup_logging():
    """設置日誌系統"""
    try:
        # 確保日誌目錄存在
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日誌
        logger.remove()  # 移除默認處理程序
        logger.add(
            sys.stderr,
            level=LOGGING_CONFIG['level']
        )
        logger.add(
            LOGGING_CONFIG['log_file'],
            rotation=LOGGING_CONFIG['rotation'],
            retention=LOGGING_CONFIG['retention'],
            level=LOGGING_CONFIG['level']
        )
    except Exception as e:
        print(f"設置日誌系統失敗: {str(e)}")
        sys.exit(1)

def load_models(model_dir: str) -> Dict[str, Any]:
    """
    加載訓練好的模型
    
    參數:
        model_dir: 模型目錄
        
    返回:
        Dict[str, Any]: 模型字典
    """
    try:
        # TODO: 實現模型加載邏輯
        return {}
    except Exception as e:
        logger.error(f"加載模型失敗: {str(e)}")
        return {}

def scanner(data_service: DataService, trading_service: TradingService, models: Dict[str, Any]):
    """
    掃描市場並執行交易
    
    參數:
        data_service: 數據服務實例
        trading_service: 交易服務實例
        models: 模型字典
    """
    try:
        # 獲取股票列表
        stock_list = data_service.load_stock_list()
        if not stock_list:
            logger.warning("股票列表為空")
            return
            
        logger.info(f"開始掃描 {len(stock_list)} 支股票")
        
        # 檢查現有倉位
        positions = trading_service.get_position_summary()
        if not positions.empty:
            logger.info(f"當前持倉:\n{positions}")
            
        # 掃描每支股票
        for symbol in stock_list:
            try:
                # 獲取最新數據
                df = data_service.get_stock_data(symbol, period="5d")
                if df is None or df.empty:
                    continue
                    
                # 添加技術指標
                df = data_service.add_technical_indicators(df)
                
                # 準備特徵
                features, _ = data_service.prepare_features(df)
                
                # TODO: 使用模型進行預測
                # predictions = make_predictions(models, features)
                
                # 檢查止損止盈
                if symbol in trading_service.positions:
                    current_price = df['Close'].iloc[-1]
                    action = trading_service.check_stop_loss_take_profit(symbol, current_price)
                    if action == 'SELL':
                        trading_service.execute_sell(symbol, current_price)
                
                # TODO: 根據預測執行交易
                # execute_trades(trading_service, predictions)
                
            except Exception as e:
                logger.error(f"處理股票失敗 {symbol}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"掃描過程發生錯誤: {str(e)}")

def main():
    """主函數"""
    try:
        # 設置日誌
        setup_logging()
        logger.info("啟動自動交易系統")
        
        # 加載配置
        config = get_all_config()
        
        # 初始化服務
        data_service = DataService(config['data'])
        trading_service = TradingService(config['trading'])
        
        # 加載模型
        models = load_models(config['paths']['model'])
        
        # 設置定時任務
        schedule.every().day.at(config['trading']['schedule_time']).do(
            scanner, data_service, trading_service, models
        )
        
        logger.info(f"系統已設置，將在每天 {config['trading']['schedule_time']} 執行掃描")
        
        # 主循環
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("收到中斷信號，正在關閉系統...")
                break
            except Exception as e:
                logger.error(f"執行定時任務時發生錯誤: {str(e)}")
                time.sleep(60)  # 發生錯誤時等待一分鐘再繼續
                
    except Exception as e:
        logger.error(f"系統啟動失敗: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()