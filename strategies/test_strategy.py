
# 初始化你嘅交易策略
class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    
    def next(self):
        # 簡單嘅交易邏輯：當收盤價大於開盤價時買入，小於開盤價時賣出
        if self.data.close[0] > self.data.open[0]:
            self.buy(size=1)
        elif self.data.close[0] < self.data.open[0]:
            self.sell(size=1)