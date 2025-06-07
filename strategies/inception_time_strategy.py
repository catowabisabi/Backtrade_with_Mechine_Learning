import backtrader as bt
import torch
import numpy as np
from typing import Dict, Any
import pandas as pd

class StockInceptionTime(torch.nn.Module):
    """InceptionTime model for stock prediction"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 定義模型層
        self.inception_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        
        for block in self.inception_blocks:
            x = block(x)
            
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

class InceptionTimeStrategy(bt.Strategy):
    """
    使用InceptionTime深度學習模型的交易策略
    """
    
    params = (
        ('window_size', 10),  # 用於預測的時間窗口大小
        ('threshold', 0.5),   # 交易信號閾值
        ('debug', False),     # 是否打印調試信息
    )

    def __init__(self, model_path: str, model_params: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            model_path: 模型權重文件路徑
            model_params: 模型參數字典
        """
        self.model = StockInceptionTime(
            input_dim=model_params['input_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            num_classes=model_params['num_classes'],
            dropout=model_params.get('dropout', 0.2)
        )
        
        # 加載模型權重
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 初始化數據存儲
        self.data_window = []
        self.order = None
        
        # 獲取特徵列表
        self.feature_names = self.get_feature_names()

    def log(self, txt: str, dt=None):
        """日誌功能"""
        if self.params.debug:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def get_feature_names(self) -> list:
        """獲取用於預測的特徵名稱列表"""
        # 這裡需要根據實際數據來定義特徵
        return ['Open', 'High', 'Low', 'Close', 'Volume']  # 示例特徵

    def prepare_data(self) -> torch.Tensor:
        """準備模型輸入數據"""
        if len(self.data_window) < self.params.window_size:
            return None
            
        # 轉換為numpy數組
        data_array = np.array(self.data_window[-self.params.window_size:])
        
        # 標準化數據（這裡使用簡單的min-max標準化，實際應該使用與訓練時相同的方法）
        data_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0) + 1e-8)
        
        # 轉換為PyTorch張量
        return torch.FloatTensor(data_normalized).unsqueeze(0)  # 添加batch維度

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
        使用深度學習模型進行預測並執行交易
        """
        # 如果有待處理的訂單，不執行新的交易
        if self.order:
            return

        # 收集當前時間點的特徵數據
        current_features = [getattr(self.datas[0], feature)[0] for feature in self.feature_names]
        self.data_window.append(current_features)

        # 準備模型輸入數據
        model_input = self.prepare_data()
        if model_input is None:
            return

        # 使用模型進行預測
        with torch.no_grad():
            prediction = self.model(model_input)
            probability = torch.sigmoid(prediction).item()  # 假設是二分類問題

        self.log(f'預測概率: {probability:.4f}')

        # 根據預測結果執行交易
        if not self.position:  # 沒有持倉
            if probability > self.params.threshold:
                self.log(f'買入信號, 概率: {probability:.4f}')
                self.order = self.buy()
        else:  # 有持倉
            if probability < (1 - self.params.threshold):
                self.log(f'賣出信號, 概率: {probability:.4f}')
                self.order = self.sell()