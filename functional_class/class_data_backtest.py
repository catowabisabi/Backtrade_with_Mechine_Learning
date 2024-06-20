import torch
import numpy as np

class ModelBacktester:
    def __init__(self, models_list):
        self.models_list = models_list

    def backtest_model(self, model, data):
        # 假設data是一個包含特徵和目標值的tuple
        features, targets = data
        model.eval()
        with torch.no_grad():
            predictions = model(features)
            # 計算回測結果，這裡只是一個示例
            accuracy = (predictions.argmax(1) == targets).float().mean()
        return accuracy.item()

    def batch_backtest(self, data):
        results = []
        for folder_models in self.models_list:
            folder_results = []
            for model in folder_models:
                result = self.backtest_model(model, data)
                folder_results.append(result)
            results.append(folder_results)
        return results
    








import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

class TestProcess:
    def __init__(self, model, dataset, batch_size=32, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def run_test(self, pth_file, output_folder):
        # 準備數據
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # 將模型設置為評估模式
        self.model.to(self.device)
        self.model.eval()

        # 創建輸出文件
        output_file = os.path.join(output_folder, os.path.basename(pth_file).replace('.pth', '.txt'))
        with open(output_file, 'w') as f:
            f.write(f'{os.path.basename(pth_file)}\n')
            
            # 測試模型
            with torch.no_grad():
                for batch_idx, (features, targets) in enumerate(data_loader):
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    
                    # 打印和保存每個批次的結果
                    f.write(f'Batch {batch_idx + 1}:\n')
                    for output, target in zip(outputs, targets):
                        f.write(f'Output: {output}, Target: {target}\n')
                    f.write('\n')

# 假設有一個模型和數據集，並初始化TestProcess
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 示例特徵和目標
features = torch.randn(100, 10)  # 100個樣本，每個樣本有10個特徵
targets = torch.randint(0, 2, (100,))  # 100個目標

# 初始化數據集
dataset = SimpleDataset(features, targets)

# 初始化模型（假設你的模型已經定義好並加載了權重）
input_dim = 10
hidden_dim = 64
num_layers = 3
num_classes = 2
model = StockInceptionTime(input_dim, hidden_dim, num_layers, num_classes)

# 假設模型權重已加載
# model.load_state_dict(torch.load('path_to_model_weights.pth'))

# 初始化測試過程
test_process = TestProcess(model, dataset, batch_size=16, device='cpu')

# 運行測試並保存結果
test_process.run_test('path_to_model_weights.pth', 'output_folder_path')







# 假設有一些虛構的測試數據
dummy_features = torch.randn(100, 1, 51)  # 100個樣本, 1個時間序列深度, 51個特徵
dummy_targets = torch.randint(0, 10, (100,))  # 100個樣本, 10個類別
dummy_data = (dummy_features, dummy_targets)

# 初始化回測器
backtester = ModelBacktester(inc_model_list)

# 批量回測所有資料夾中的模型
results = backtester.batch_backtest(dummy_data)

for folder_idx, folder_results in enumerate(results):
    print(f"Testing models in Folder {folder_idx}:")
    for model_idx, result in enumerate(folder_results):
        print(f"  Model {model_idx}: Accuracy = {result:.2%}")


