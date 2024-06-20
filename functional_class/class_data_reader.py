import pandas as pd



# 單 1 csv 變為 單一df
class DataReader:

    def __init__(self, data_csv_path):
        self.data_csv_path = data_csv_path
        print(f"回測所用的資料由這文件讀取: {data_csv_path}")
        self.df = None

    # 讀取文件為pandas dataframe
    def read_csv(self):
        self.df = pd.read_csv(self.data_csv_path)
        print("數據讀取完畢！")

    # 打印頭和尾10行
    def print_df(self):
        if self.df is not None:
            print("數據頭10行：")
            print(self.df.head(10))
            print("數據尾10行：")
            print(self.df.tail(10))
        else:
            print("數據未讀取，請先調用 read_csv 方法。")

    # 從dataframe中移除不需要的特徵列
    def remove_features_from_df(self, list_of_features):
        self.df = self.df.drop(columns=list_of_features, errors='ignore')
        print(f"已移除特徵：{list_of_features}")

    def date_to_datetime_and_index(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)


    def run(self):
        self.read_csv()
        #self.print_df()
        self.remove_features_from_df(["PREDICT_TARGET"])
        self.date_to_datetime_and_index()
        #self.print_df()
        return self.df