import backtrader as bt


class BuyHold(bt.Strategy):
    """
    簡單的買入並持有策略
    在第一個交易日買入所有可能的股票，然後一直持有到回測結束
    """
    
    params = (
        ('debug', False),  # 是否打印調試信息
    )

    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.order = None  # 追踪訂單
        self.bought = False  # 是否已經買入

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
            self.bought = False  # 重置買入標誌

        self.order = None

    def next(self):
        """
        主要策略邏輯
        只在開始時買入一次，之後一直持有
        """
        # 如果已經有訂單或已經買入，不執行任何操作
        if self.order or self.bought:
            return

        # 計算可以買入的股票數量
        cash = self.broker.getcash()
        price = self.dataclose[0]
        if price > 0:  # 防止除以零
            size = int(cash / price)
            if size > 0:
                self.log(f'買入訂單, 價格: {price:.2f}, 數量: {size}')
                self.order = self.buy(size=size)
                self.bought = True  # 標記已經買入