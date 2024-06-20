import torch
from models.model_define.InceptionTime.train_class_inception_time_v2 import StockInceptionTime
import torch.nn as nn
class ModelInitializer:
    def __init__(self, settings_list, model_folder_list):
        self.settings_list = settings_list
        self.model_folder_list = model_folder_list
        self.inc_model_list = []

    def initialize_models(self):
        for setting, model_files in zip(self.settings_list, self.model_folder_list):
            folder_models = []

            for model_file in model_files:
                # 假設設定檔中包含初始化所需的參數
                input_dim = setting.get('input_dim', 1)
                hidden_dim = setting.get('hidden_dim', 64)
                num_layers = setting.get('num_layers', 3)
                num_classes = setting.get('num_classes', 10)
                dropout = setting.get('dropout', 0.1)
                l1_lambda = setting.get('l1_lambda', 0.0)
                l2_lambda = setting.get('l2_lambda', 0.0)

                # 初始化模型
                model = StockInceptionTime(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    dropout=dropout,
                    l1_lambda=l1_lambda,
                    l2_lambda=l2_lambda
                )

                # 加載模型權重
                model.load_state_dict(torch.load(model_file))
                folder_models.append(model)

            self.inc_model_list.append(folder_models)
        #self.show_model_loaded()

    def show_model_loaded(self):
        # 打印結果
        print("Initialized Models:")
        for folder_idx, models in enumerate(self.inc_model_list):
            print(f"Folder {folder_idx}:")
            for model_idx, model in enumerate(models):
                print(f"  Model {model_idx}: {model}")

    def show_model_count(self):
        # 打印每個文件夾中模型的數量
        print("Initialized Models:")
        for folder_idx, models in enumerate(self.inc_model_list):
            print(f"Folder {folder_idx}: Number of models = {len(models)}")

    def get_models(self):
        print("得到 一個 文件清單 中的 模型清單 [[model1 model2], [model3 model4 model4]...]")
        self.show_model_count()
        return self.inc_model_list