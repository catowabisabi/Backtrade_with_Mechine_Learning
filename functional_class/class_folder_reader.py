import os
import pandas as pd

default_ta_folder_path = "Y:/TrueNas_4TB_A/Coding Share/ta_data/test"


class FolderReader:
    def __init__(self, folder_path = default_ta_folder_path, ext = "test"):
        self.folder_path = folder_path
        self.ext         = f"_{ext}"

    def read_csv_files(self):
        dataframes = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(f'{self.ext}.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
        #for df in dataframes:
        print(df.columns)
        print(dataframes[0])

        return dataframes