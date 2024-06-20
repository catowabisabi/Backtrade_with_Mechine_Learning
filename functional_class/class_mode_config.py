from models.model_define.InceptionTime.train_class_inception_time_v2 import StockInceptionTime
import torch

class ModelLoader:
    def __init__(self, model_path, settings):
        # 將settings字典中的每個鍵值對設置為類的屬性
        for key, value in settings.items():
            setattr(self, key, value)

        self.model = StockInceptionTime(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, num_classes=self.num_classes, dropout=self.dropout, l1_lambda=self.l1_lambda, l2_lambda=self.l2_lambda)
        self.model.load_state_dict(torch.load(model_path))


    def load_eval_model(self):
        self.model.eval()

    def predict(self, input):
        with torch.no_grad():
            prediction = self.model(input)
            return prediction

    def test(self, dataset, targets):
        predictions = self.model(dataset)
        result_1 = predictions[:, 0] / targets[:, 0]
        result_2 = predictions[:, 1] / targets[:, 1]
        result_3 = predictions[:, 2] / targets[:, 2]

        return {
            'percentage_0': result_1.mean().item(),
            'percentage_1': result_2.mean().item(),
            'percentage_2': result_3.mean().item()
        }

    def run_test(self):
        # 此方法需要具體實現
        pass

    def run_prediction_tomorrow(self):
        # 此方法需要具體實現
        pass
