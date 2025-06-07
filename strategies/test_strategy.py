import backtrader as bt

# 初始化你嘅交易策略
class TestStrategy(bt.Strategy):
    """
    A simple test strategy that demonstrates basic Backtrader functionality.
    Buy when close price is higher than open price, sell when it's lower.
    """
    
    params = (
        ('size', 1),  # 交易數量
        ('debug', False),  # 是否打印調試信息
    )

    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.order = None  # 追踪訂單

    def log(self, txt, dt=None):
        """日誌功能"""
        if self.params.debug:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """訂單狀態通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'買入執行, 價格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手續費: {order.executed.comm:.2f}')
            else:
                self.log(f'賣出執行, 價格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手續費: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單取消/保證金不足/拒絕')

        self.order = None

    def next(self):
        """
        主要策略邏輯
        當收盤價大於開盤價時買入，小於開盤價時賣出
        """
        self.log(f'收盤價: {self.dataclose[0]:.2f}, 開盤價: {self.dataopen[0]:.2f}')

        # 檢查是否有待處理訂單
        if self.order:
            return

        # 檢查是否持倉
        if not self.position:
            # 沒有持倉 - 考慮買入
            if self.dataclose[0] > self.dataopen[0]:
                self.log(f'買入信號, 價格: {self.dataclose[0]:.2f}')
                self.order = self.buy(size=self.params.size)
        else:
            # 有持倉 - 考慮賣出
            if self.dataclose[0] < self.dataopen[0]:
                self.log(f'賣出信號, 價格: {self.dataclose[0]:.2f}')
                self.order = self.sell(size=self.params.size)