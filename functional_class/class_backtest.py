
from strategies.golden_cross import GoldenCross
from functional_class.class_data_reader import DataReader ## 單1 DF
from config.config import features_0, features_1, features_2, features_3, features_5
import backtrader as bt

# import the package after installation
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import plotly.io


class Backtest:
    def __init__(self, ticker="CHTR", slp=2, tpp=2, com=0.1):
        self.ticker = ticker
        self.slp = slp
        self.tpp = tpp
        self.com = com
     
       

    def get_df(self):
        dr = DataReader(f'predict_csv/0_{self.ticker}_predict.csv')
        stock_ta_df = dr.run()
        return stock_ta_df

    def run(self):
        data = ExtendedPandasData(dataname=self.get_df())

        # 初始化回測引擎
        cerebro = bt.Cerebro()
        cerebro.addstrategy(GoldenCross)
        cerebro.adddata(data)
        # 设置交易佣金为 0.2%
        cerebro.broker.setcommission(commission=0.001)  # 0.2% 的佣金
        cerebro.broker.setcash(1000.0)

        # 執行回測
        results = cerebro.run()

        # define plot scheme with new additional scheme arguments
        scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)

        figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))

        # directly manipulate object using methods provided by `plotly`
        for i, each_run in enumerate(figs):
            for j, each_strategy_fig in enumerate(each_run):
                # open plot in browser
                each_strategy_fig.show()

                # save the html of the plot to a variable
                html = plotly.io.to_html(each_strategy_fig, full_html=False)

                # write html to disk
                plotly.io.write_html(each_strategy_fig, f'{i}_{j}.html', full_html=True)

class ExtendedPandasData(bt.feeds.PandasData):
    lines = tuple(features_3)  # 將 features_0 轉換為元組，並用作 lines 的定義
    params = {feature: -1 for feature in features_3}  # 使用字典推導式設置每個特徵的參數
    



