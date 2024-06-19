import os
import backtrader as bt
import yfinance as yf
import pandas as pd

def get_model_paths(models_folder):
    model_paths = []
    for root, dirs, files in os.walk(models_folder):
        for file in files:
            if file.endswith(".pth"):
                model_paths.append(os.path.join(root, file))
    return model_paths

def load_data(stock_list_file):
    stock_list = pd.read_csv(stock_list_file)
    data = {}
    for stock in stock_list['Symbol']:
        data[stock] = yf.download(stock, start="2010-01-01", end="2023-06-12")
    return data

def backtest(model_path, data, params):
    # 使用指定的模型和參數對數據進行回測
    # 返回回測結果
    pass

def main():
    models_folder = "models"
    stock_list_file = "stock_list.csv"
    params = {...}  # 指定回測參數

    model_paths = get_model_paths(models_folder)
    data = load_data(stock_list_file)

    results = {}
    for model_path in model_paths:
        model_results = {}
        for stock, stock_data in data.items():
            result = backtest(model_path, stock_data, params)
            model_results[stock] = result
        results[model_path] = model_results

    # 分析和總結回測結果
    pass

if __name__ == "__main__":
    main()