import schedule
import time
import yfinance as yf
import pandas as pd

def load_models(models_folder):
    # 加載訓練好的模型
    pass

def load_stock_list(stock_list_file):
    stock_list = pd.read_csv(stock_list_file)
    return stock_list['Symbol'].tolist()

def get_latest_data(stock_list):
    # 獲取股票列表中每隻股票的最新數據
    pass

def make_predictions(models, data):
    # 使用加載的模型對最新數據進行預測
    pass

def execute_trades(predictions):
    # 根據預測結果執行買賣操作
    pass

def scanner():
    models = load_models("models")
    stock_list = load_stock_list("stock_list.csv")
    data = get_latest_data(stock_list)
    predictions = make_predictions(models, data)
    execute_trades(predictions)

def main():
    schedule.every().day.at("09:00").do(scanner)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()