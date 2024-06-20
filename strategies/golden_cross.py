import math
import backtrader as bt
import pandas as pd
import time
from config.config import features_0, features_1, features_2, features_3, features_5

class GoldenCross(bt.Strategy):

    params = (('fast', 50) , 
              ('slow', 200) , 
              ('order_percentage', 0.95) ,
              ('ticker','SPY'),
              ('slp',1.5),
              ('tpp',1),
              ('com',0.1)
              )

    
    def __init__(self):
        
        self.fast_moving_average = bt.indicators.SMA(
            self.data.close, period = self.params.fast,
            plotname = '50 days moving average'
        )

        self.slow_moving_average = bt.indicators.SMA(
            self.data.close, period = self.params.slow,
            plotname = '200 days moving average'
        )

        self.crossover = bt.indicators.CrossOver(self.fast_moving_average, self.slow_moving_average)
        self.data_log = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] + features_3)

        self.buy_price = None  # 初始化買入價格為None

        


 

    def next(self):

        signal = None
         # 获取日期
        date = self.data.datetime.date(0)
        
        # 创建当天数据的字典，确保所有值都被封装在列表中
        today_data = {
            'Date': [date],  # 使用列表包装单个日期值
            'Open': [self.data.open[0]],
            'High': [self.data.high[0]],
            'Low': [self.data.low[0]],
            'Close': [self.data.close[0]],
            'Volume': [self.data.volume[0]]
        }

        # 添加其他特征
        for feature in features_3:
            today_data[feature] = [getattr(self.data, feature)[0]] if hasattr(self.data, feature) else ['Not available']

        # 使用 pd.concat 来添加当天数据到 DataFrame
        new_entry = pd.DataFrame(today_data)
        self.data_log = pd.concat([self.data_log, new_entry], ignore_index=True)
        
        # 當 DataFrame 長度達到 10 天時, 使用滾動窗口
        if len(self.data_log) >= 10:
            windowed_data = self.data_log.tail(10)  # 獲取最後 10 天的數據
            signal = self.calculated_signal(windowed_data)  # 假設已定義此函數來計算基於窗口的信號
        
            #print(windowed_data)

                



        # 當前沒有持倉，檢查是否應該買入
        if self.position.size == 0  :
            if signal == 1:
                #if  self.fast_moving_average > self.slow_moving_average:
                    amount_to_invest = (self.params.order_percentage * self.broker.cash)
                    self.size = math.floor(amount_to_invest / self.data.close[0])  # 確保使用當前的收盤價

                    #print("買入 {} 股 {} 於價格 {}".format(self.size, self.params.ticker, self.data.close[0]))
                    """ print("開盤價: {}, 最高價: {}, 最低價: {}, 收盤價: {}".format(
                        self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0])) """
                    
                    self.buy(size=self.size)  # 執行買入操作
                    self.buy_price = self.data.close[0]  # 記錄買入價格


         # 已有持倉，檢查是否應該賣出
        if self.position.size > 0:
            current_price = self.data.close[0]
            pct_change = (current_price - self.buy_price) / self.buy_price  # 使用記錄的買入價格計算pct_change


            #if signal == -1 or pct_change >= 0.015 or pct_change <= -0.01:
            if signal == -1 or (pct_change >= (self.params.tpp/100)) or (pct_change <= -(self.params.slp/100)):
                #print("賣出 {} 股 {} 於價格 {}".format(self.size, self.params.ticker, self.data.close[0]))
                self.close()  # 執行賣出操作，關閉持倉
        




    def calculated_signal(self, windowed_data):
        #print(windowed_data)
        # 确保 DataFrame 中有 'DIRECTION' 列
        if 'Prediction' in windowed_data.columns:
            # 获取 'DIRECTION' 列的最后一个值
            signal = windowed_data['Prediction'].iloc[-1]
        else:
            print('NO Prediction')
            # 如果没有 'DIRECTION' 列，可以设置一个默认信号或处理错误
            signal = -1  # 或者你可以设定一个默认值或抛出异常

        
        return signal