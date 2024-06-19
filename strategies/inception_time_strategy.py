




class InceptionTimeStrategy(bt.Strategy):
    def __init__(self, model_path, params):
        self.model = StockInceptionTime(params['input_dim'], params['hidden_dim'], params['num_layers'], params['num_classes'], params['dropout'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


        

    def next(self):
        # 使用模型對當前數據進行預測
        # 根據預測結果做出交易決策
        pass