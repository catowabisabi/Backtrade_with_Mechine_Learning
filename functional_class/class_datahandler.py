import os
import yfinance as yf
import pandas as pd
import csv
import numpy as np
import talib
import yfinance as yf


class DataHandler:
    def __init__(self):

        self.stock_list_csv  = "data_prep\stock_list\source\SP500.csv"
        self.output_folder   = 'data_prep\stock_list\sectors'
        self.history_folder  = 'data_prep\historial_data'
        self.download_period = 'max'
        self.download_interval='1D'
        self.ta_data_folder = '../data_prep/ta_data'

    def list_sector_folders(self):
        for folder_name in os.listdir(self.history_folder):
            print(folder_name)

    def download(self):
        #data = DataHandler.read_csv(self.stock_list_csv)
        self.create_sector_files()
        os.makedirs(self.history_folder, exist_ok=True)
        self.download_stock_data()



    def read_csv(self):
        data = pd.read_csv(self.stock_list_csv)
        print(data.head())
        return data


    def create_sector_files(self):
        # 創建輸出文件夾(如果不存在)
        os.makedirs(self.output_folder, exist_ok=True)

        # 讀取輸入的CSV文件
        with open(self.stock_list_csv, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳過標題行
            
            # 創建一個字典來存儲每個行業的股票代碼
            sectors = {}
            
            # 遍歷CSV文件的每一行
            for row in reader:
                symbol, name, sector = row
                
                # 將股票代碼添加到對應的行業列表中
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(symbol)
        
        # 為每個行業創建一個CSV文件
        for sector, symbols in sectors.items():
            sector_file = f"{self.output_folder}/{sector}.csv"
            with open(sector_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([[symbol] for symbol in symbols])
        
        
    def download_stock_data(self):
    # 遍歷sectors文件夾中的每個行業文件
    
        for sector_file in os.listdir(self.output_folder):
            sector = os.path.splitext(sector_file)[0]
            
            # 創建行業文件夾(如果不存在)
            sector_folder = f"{self.history_folder}/{sector}"
            os.makedirs(sector_folder, exist_ok=True)
            print(sector_folder)
            
            # 讀取行業文件中的股票代碼
            with open(f"{self.output_folder}/{sector_file}", 'r') as file:
                symbols = [line.strip() for line in file]
            
            # 下載每隻股票的1小時數據
            for symbol in symbols:
                stock_data = yf.download(symbol, period=self.download_period, interval=self.download_interval)
                
                # 將數據保存到CSV文件
                output_folder = f"{sector_folder}/{symbol}"
                os.makedirs(output_folder, exist_ok=True)
                print(output_folder)


                output_file = f"{output_folder}/{symbol}_{self.download_interval}.csv"
                print(output_file)
                stock_data.to_csv(output_file)
                
                print(f"Downloaded data for {symbol} in {sector} sector.")















class TaHandler:
    def __init__(self):
        self.history_folder = 'data_prep/historial_data'
        self.cleanup_data_folder = self.history_folder

        self.download_period = 'max'
        self.download_interval='1D'

        #最後要保存的文件夾
        self.ta_data_folder = 'data_prep/ta_data'

        #df = pd.read_csv(f'{self.history_folder}/Consumer Discretionary/AAP/AAP_1D.csv')
    
    def list_sector_folders(self):
        for folder_name in os.listdir(self.history_folder):
            print(folder_name)
                
        
    @staticmethod
    def make_folder(path):
        # 檢查文件夾是否存在
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f'Folder created: {path}')
        


    def create_paths(self, folder_path, file_name, sub_folder_name, stock_folder_name):
        csv_file_path = os.path.join(folder_path, file_name) # 得到所有csv , yf 下載的原始東西
        #print(f"yf 下載的CSV :  {csv_file_path}")

        ta_csv_save_folder = os.path.join(self.ta_data_folder, sub_folder_name)
        self.make_folder(ta_csv_save_folder)
        ta_csv_save_folder = os.path.join(ta_csv_save_folder, stock_folder_name)
        self.make_folder(ta_csv_save_folder)
        ta_csv_filename = f"{stock_folder_name.split('.')[0]}_1D_ta.csv"

        #print(f"要保存ta CSV 的文件夾 : {ta_csv_save_folder}")
        #print(f"新的文件名 : {ta_csv_filename}")
        #print('')
        return csv_file_path, ta_csv_save_folder, ta_csv_filename




    @staticmethod
    def remove_adj(df):
        # 删除'Adj Close'列(如果存在)
        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)
        return df

    @staticmethod
    def to_num(df):
        # 将价格和成交量数据转换为数字类型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
            df[col] = df[col].astype('float64')
            
        # 将价格数据转换为小数点后两位
        price_columns = ['Open', 'High', 'Low', 'Close']
        df[price_columns] = df[price_columns].round(2)

        return df
    
    @staticmethod
    def add_ma(df):
        # 计算技术指标
        df['MA5'] = talib.SMA(df['Close'], timeperiod=5).round(2)
        df['MA6'] = talib.SMA(df['Close'], timeperiod=6).round(2)
        df['MA10'] = talib.SMA(df['Close'], timeperiod=10).round(2)
        df['MA20'] = talib.SMA(df['Close'], timeperiod=20).round(2)
        return df
        
    @staticmethod
    def add_bias(df):
        # 计算 5 日和 10 日的乖离率
        df['BIAS5'] = ((df['Close'] - df['MA5']) / df['MA5'] * 100).round(2)
        df['BIAS10'] = ((df['Close'] - df['MA10']) / df['MA10'] * 100).round(2)

        return df
    
    @staticmethod
    def add_rsi(df):
        df['RSI6'] = talib.RSI(df['Close'], timeperiod=6).round(2)
        df['RSI12'] = talib.RSI(df['Close'], timeperiod=12).round(2)
        df['RSI6_pct_change'] = (df['RSI6'].pct_change()*100).round(2)
        df['RSI12_pct_change'] = (df['RSI12'].pct_change()*100).round(2)
        df['rsi6_direction'] = np.where(df['RSI6_pct_change'] > 0, 1, np.where(df['RSI6_pct_change'] < 0, -1, 0))
        df['rsi12_direction'] = np.where(df['RSI12_pct_change'] > 0, 1, np.where(df['RSI12_pct_change'] < 0, -1, 0))
        return df

    @staticmethod
    def add_other_ta(df):
        df['Volume10'] = talib.SMA(df['Volume'], timeperiod=10).round(2)
        
        df['12W%R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=12).round(2)
        df['9K'], df['9D'] = [x.round(2) for x in talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=9)]
        df['MACD'], _, _ = [x.round(2) for x in talib.MACD(df['Close'])]
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14).round(2)
        return df

    @staticmethod
    def add_change(df):
        # 计算价格变化百分比
        df['close_pct_change'] = (df['Close'].pct_change()*100).round(2)
        df['high_pct_change'] = (df['High'].pct_change()*100).round(2)
        df['low_pct_change'] = (df['Low'].pct_change()*100).round(2)
        df['VolPctChg'] =  (df['Volume'].pct_change()*100).round(2)
        df['volume_direction'] = np.where(df['VolPctChg'] > 0, 1, np.where(df['VolPctChg'] < 0, -1, 0))
        return df

    @staticmethod
    def add_diff(df):
        # 计算技术指标的差值
        df['MA5_DIFF'] = df['MA5'].diff().round(2)
        df['MA6_DIFF'] = df['MA6'].diff().round(2)
        df['MA10_DIFF'] = df['MA10'].diff().round(2)
        df['MA20_DIFF'] = df['MA20'].diff().round(2)
        df['BIAS5_DIFF'] = df['BIAS5'].diff().round(2)
        df['BIAS10_DIFF'] = df['BIAS10'].diff().round(2)
        df['RSI6_DIFF'] = df['RSI6'].diff().round(2)
        df['RSI12_DIFF'] = df['RSI12'].diff().round(2)
        df['2W%R_DIFF'] = df['12W%R'].diff().round(2)
        df['K_DIFF'] = df['9K'].diff().round(2)
        df['D_DIFF'] = df['9D'].diff().round(2)
        df['MACD_DIFF'] = df['MACD'].diff().round(2)
        df['Volume_Diff'] = df['Volume'].diff().round(2)
        df['TREND_LINE_DIFF'] = abs(df['MA20'] - df['MA10']).round(2)
        return df

    @staticmethod
    def add_bar_ta(df):
        # 计算 body_size_to_price_prec
        df['BODY_PERC'] = (abs(df['Close'] - df['Open']) / df['Open'] * 100).round(2)
        # 计算 close-open
        df['GREEN_RED'] = np.where(df['Close'] > df['Open'], 1, np.where(df['Close'] < df['Open'], -1, 0))
        return df


        
    @staticmethod
    def add_trend(df):
        df['TREND'] = np.where(df['MA20'] > df['MA10'], -1, np.where(df['MA20'] < df['MA10'], 1, 0))
        return df

    @staticmethod
    def add_target(df):
        # 創建 'direction' 列
        df['DIRECTION'] = np.where(df['close_pct_change'] > 1, 1, np.where(df['close_pct_change'] < -1, -1, 0))
        # 創建 'PREDICT_TARGET' 列，將 'direction' 列的數值向下移動一行
        df['PREDICT_TARGET'] = df['DIRECTION'].shift(-1)
        return df
                

    @staticmethod
    def to_ta(df):
        df = TaHandler.remove_adj(df)
        df = TaHandler.to_num(df)
        df = TaHandler.add_ma(df)
        df = TaHandler.add_bias(df)
        df = TaHandler.add_rsi(df)
        df = TaHandler.add_other_ta(df)
        df = TaHandler.add_bar_ta(df)
        df = TaHandler.add_diff(df)
        df = TaHandler.add_change(df)
        df = TaHandler.add_trend(df)
        df = TaHandler.add_target(df)
        return df

    def set_index(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)  
        return df

    def get_vix2(self, df):
        
        vix = yf.Ticker("^VIX")
        vix_data= vix.history(period=self.download_period, interval=self.download_interval)

        vix_data.reset_index(inplace=True)
        vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.tz_localize(None)
        vix_data.set_index('Date', inplace=True)

        #vix_data[['Close']].rename(columns={'Close': 'VIX'})

        vix_data_renamed = vix_data[['Close']].rename(columns={'Close': 'VIX'})

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index('Date', inplace=True)

        df = df.join(vix_data_renamed, how='left')
        #df.dropna(inplace=True)
        #重置Date
        df.reset_index(inplace=True, drop=False)
        return df


    @staticmethod
    def add_indies( df, indies):
        indies.reset_index(inplace=True)
        indies['Date'] = pd.to_datetime(indies['Date']).dt.tz_localize(None)
        indies.set_index('Date', inplace=True)

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index('Date', inplace=True)

        df = df.join(indies, how='left')
        df.reset_index(inplace=True, drop=False)
        return df

    


    @staticmethod
    def save_to_csv(df, save_folder, filename):
        """将 DataFrame 保存到 CSV 文件中"""
        save_path = os.path.join(save_folder, filename)
        df.to_csv(save_path, index=False)  # 根据需要决定是否需要保存索引
        print(f"Saved: {save_path}")



    @staticmethod
    # 定義要下載的股市指數
    def get_index():
        print("hello")
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones Industrial Average": "^DJI",
            "NASDAQ Composite": "^IXIC",
            "Russell 2000": "^RUT"
        }

        # 下載股市指數數據
        index_data_frames = {name: yf.download(ticker, period = 'max')['Close']
                            for name, ticker in indices.items()}

        #fred_series = {
        #    "Unemployment Rate": "UNRATE",  # 失業率
        #    "Consumer Confidence": "UMCSENT",  # 消費者信心指數
        #    "Manufacturing PMI": "NAPM",  # 假設找到的正確的代碼
        #    "Consumer Price Index": "CPIAUCSL"  # 消費者物價指數
        #} 
        

        # 合併所有數據到一個 DataFrame
        all_data = pd.concat(index_data_frames.values(), axis=1, keys=index_data_frames.keys())


        # 處理 NaN 值，使用向前填充
        all_data.dropna(inplace=True)

        all_data = all_data.round(2)

        
        return all_data




    def process_stock_data(self, folder_path,  sub_folder_name, stock_folder_name, indices):
        #print ('開啓')
        dfs = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith("_1D.csv"):

                csv_file_path, ta_csv_save_folder, ta_csv_filename = self.create_paths(folder_path, file_name, sub_folder_name, stock_folder_name)
                print(csv_file_path)
        
                try:
                
                    df = pd.read_csv(csv_file_path)
                    #print(df)
                    df = TaHandler.to_ta(df)
                    #print(df)
                    df = self.get_vix2(df)
                    df = TaHandler.add_indies(df, indices)
                    df.dropna(inplace=True)
                    df = df.round(2)

                    # 确保保存目录存在
                    if not os.path.exists(ta_csv_save_folder):
                        os.makedirs(ta_csv_save_folder)

                    # 保存处理后的 DataFrame 到 CSV 文件

                    TaHandler.save_to_csv(df, ta_csv_save_folder, ta_csv_filename)
                    


                    # 如果删除后没有数据,则跳过该文件
                    if df.empty:
                        print(f"\nNo data for {file_name} after processing, deleting the file and folder.")
                        os.remove(csv_file_path)
                        os.rmdir(folder_path)
                        continue
                    
                    dfs.append(df)

                except Exception as e:
                    print(f"出現錯誤: processing {file_name}: {str(e)}")
                    # 如果处理过程中出错,也删除该文件和对应的资料夹
                    #os.remove(csv_file_path)
                    #os.rmdir(folder_path)
                    continue

        
                
        if dfs:  # 确保列表不为空
            #df = dfs[0].dropna()
            return dfs
            
            #print(df.tail(20))
            #print(dfs[0].dropna().tail(20))
        else:
            print("No processed data available.")

            return None
        

    def run_gen_ta(self):
        #print("yo")
        indices = TaHandler.get_index()
        #print(indices)

        for sub_folder_name in os.listdir(self.history_folder):
            sub_folder_path = os.path.join(self.history_folder, sub_folder_name) # history folder內的所有子文件夾
            print(sub_folder_path)
            

            for stock_folder_name in os.listdir(sub_folder_path):
                stock_folder_path = os.path.join(sub_folder_path, stock_folder_name)# history folder內的所有子文件夾內的股票文件夾
                #print(stock_folder_path)
                self.process_stock_data(stock_folder_path, sub_folder_name, stock_folder_name, indices)
        
        #len(indices)


