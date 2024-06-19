
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from fc.train_cm import CM

#from fc.train_class_inception_time import StockInceptionTime
from fc.train_class_inception_time_v2 import StockInceptionTime
from datetime import datetime

class StockTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True  # 設置這個參數為 True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x

# 無windows
class StockLSTM0(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(StockLSTM, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.output_layer(x[:, -1, :])  # 取序列的最後一個時間步
        return x

    

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


    
class Trainer:

        
    # 训练模型
    def train(model, dataloader, criterion, optimizer, device, num_classes):

        model.train()
        total_loss = 0
        total_counts = torch.zeros(num_classes, device=device)  # 用於存儲每個類別的總樣本數
        correct_counts = torch.zeros(num_classes, device=device)  # 用於存儲每個類別的正確預測數

        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, l1_reg, l2_reg, l1_lambda, l2_lambda= model(features)

            loss = criterion(outputs, targets)

            # 添加 L1 和 L2 正则化项到损失函数
            loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 更新準確率計算
            _, predictions = torch.max(outputs, 1)
            correct = (predictions == targets)
            for i in range(num_classes):
                total_counts[i] += (targets == i).sum().item()
                correct_counts[i] += (correct & (targets == i)).sum().item()

        average_loss = total_loss / len(dataloader)
        total_accuracy = correct_counts.sum().item() / total_counts.sum().item()
        class_accuracies = (correct_counts / total_counts).tolist()  # 計算每個類的準確率

        return average_loss, total_accuracy, class_accuracies
    
    def calculate_accuracy(outputs, targets):
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total

    # 评估模型
    def evaluate(model, dataloader, criterion, device, num_classes):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        total_counts = torch.zeros(num_classes)  # 存儲每個類別的總數
        correct_counts = torch.zeros(num_classes)  # 存儲每個類別的正確預測數

        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(device), targets.to(device)
                outputs, l1_reg, l2_reg, l1_lambda, l2_lambda = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                correct = (predictions == targets)
                for i in range(num_classes):
                    total_counts[i] += (targets == i).sum().item()
                    correct_counts[i] += (correct & (targets == i)).sum().item()

        class_accuracies = (correct_counts / total_counts).tolist()  # 计算每个类的准确率
        total_accuracy = correct_counts.sum().item() / total_counts.sum().item()

        return total_loss / len(dataloader), total_accuracy, class_accuracies
    


    def train_and_evaluate(train_dataset, val_dataset, test_dataset,
            
        input_dim = 3,  # 特征维度
        hidden_dim = 64,
        num_layers = 8,
        num_heads = 32 ,
        num_classes = 3,
        batch_size = 32,
        num_epochs = 100,
        
        learning_rate = 0.0001, dropout = 0.1,checkpoint_path = None, mode = "LSTM", my_min_acc = 0.40,
        l1_lambda=0.0, 
        l2_lambda=0.0
        ):

        from fc.data_converter import DataCovert
        #DataCovert.check_label_num(train_dataset)
        #DataCovert.check_label_num(val_dataset)

        # 调整数据形状
       # train_dataset.features = train_dataset.features.transpose(1, 2)
       # val_dataset.features = val_dataset.features.transpose(1, 2)
       # test_dataset.features = test_dataset.features.transpose(1, 2)
    
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader      = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)

        current_time    = datetime.now().strftime("%Y%m%d_%H%M%S")
        




        # 初始化模型
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if mode == "INCEPTION":
            model       = StockInceptionTime    (input_dim, hidden_dim, num_layers, num_classes, dropout, l1_lambda, l2_lambda).to(device)
        elif mode == "LSTM":
            model       = StockLSTM             (input_dim, hidden_dim, num_layers, num_classes, dropout).to(device)
        else:
            model       = StockTransformer      (input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout).to(device)
        
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path))  # 加載已保存的模型權重
        
        criterion   = nn.CrossEntropyLoss()
        #optimizer   = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Adding weight decay for L2 regularization

        # 記錄最佳準確度
        best_class_accuracy = [0.0] * num_classes  # 紀錄每個類別的最佳準確度
        best_overall_accuracy = 0.0  # 紀錄整體的最佳準確度



        model_dir = f'model/{current_time}_best_id{input_dim}_hd{hidden_dim}_ly{num_layers}_he{num_heads}_bs{batch_size}_ne{num_epochs}_lr{learning_rate}_dp{dropout}_md{mode}_mc{my_min_acc}_l1{l1_lambda}_l2_{l2_lambda}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)


        if mode == "LSTM":
            runs = f"runs/stock_lstm_experiment_{current_time}"
        elif mode == "INCEPTION":
            runs = f"runs/stock_inception_experiment_{current_time}"
        else:
            runs = f"runs/stock_transformer_experiment_{current_time}"

        writer = SummaryWriter(runs)

        model_name = ""
        for epoch in range(num_epochs):
           

            train_loss, train_accuracy, train_class_accuracies = Trainer.train(model, train_loader, criterion, optimizer, device, num_classes)
            
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Train/DN', train_class_accuracies[0], epoch)
            writer.add_scalar('Accuracy/Train/FLAT', train_class_accuracies[1], epoch)
            writer.add_scalar('Accuracy/Train/UP', train_class_accuracies[2], epoch)

            val_loss, val_accuracy,  val_class_accuracies = CM.evaluate_with_cm(model, val_loader, criterion, device, ["DN", "NO", "UP"], epoch, model_dir = model_dir)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            writer.add_scalar('Accuracy/Validation/DN', val_class_accuracies[0], epoch)
            writer.add_scalar('Accuracy/Validation/FLAT', val_class_accuracies[1], epoch)
            writer.add_scalar('Accuracy/Validation/UP', val_class_accuracies[2], epoch)
        
            features, targets = next(iter(val_loader))
            features, targets = features.to(device), targets.to(device)
            outputs, l1_reg, l2_reg, _, _ = model(features)
            _, predictions = torch.max(outputs, 1)

            
            #print(train_accuracy)
            #print(train_class_accuracies)

            def print_acc(epoch, num_epochs, loss, accuracy, class_accuracies, name = "訓練集"):
                print(f"第 [{epoch+1}/{num_epochs}]訓練, {name} 殘差值: {loss:.4f}, {name} 準確值: {accuracy:.2%}, \
                    跌巿準確 : {class_accuracies[0]:.2%}, 橫盤準確 : {class_accuracies[1]:.2%}, 升巿準確 : {class_accuracies[2]:.2%}"  )
                
            print_acc(epoch,num_epochs,train_loss,  train_accuracy, train_class_accuracies, "訓練集")
            print_acc(epoch,num_epochs,val_loss,    val_accuracy,   val_class_accuracies,   "評估集")
        
            if mode == "LSTM":
                ext = "_lstm"
            elif mode == "INCEPTION":
                ext = "_inception"
            else:
                ext = "_transformer"

            min_acc = my_min_acc
            formatted_accuracy = f"_epoch_{epoch}_all_acc_{val_accuracy:.2f}_DN_acc_{val_class_accuracies[0]:.2%}_FLAT_acc_{val_class_accuracies[1]:.2%}_UP_acc_{val_class_accuracies[2]:.2%}_min_acc_{min_acc*100}"


        
            # 保存最佳模型
            if val_accuracy > best_overall_accuracy and val_accuracy > min_acc:
                if all(acc > min_acc for acc in val_class_accuracies):  # 檢查所有類別的準確率是否都超過40%
                    best_overall_accuracy = val_accuracy

                    model_path = os.path.join(model_dir, f'overall_best_model{ext}{formatted_accuracy}.pth')
                    model_name = model_path
                    torch.save(model.state_dict(), model_path)
                    print(f"模型保存於: {model_path}")
                #else:
                    #print("所有類別的準確率未全部超過40%,不保存模型")

            for i in range(num_classes):
                if val_class_accuracies[i] > best_class_accuracy[i] and val_class_accuracies[i] > min_acc:
                    if all(acc > min_acc for acc in val_class_accuracies):  # 檢查所有類別的準確率是否都超過40%
                        best_class_accuracy[i] = val_class_accuracies[i]
                        model_path = os.path.join(model_dir, f'{i}_best_model{ext}{formatted_accuracy}.pth')
                        torch.save(model.state_dict(), model_path)
                        print(f"模型保存於: {model_path}")
                    #else:
                        #print(f"所有類別的準確率未全部超過40%,不保存類別{i}的最佳模型")

            

        writer.close()
        
        Trainer.test_model(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout, batch_size, test_dataset, targets, epoch, mode, model_name,  model_dir = model_dir)

        
    def test_model(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout, batch_size, test_dataset, targets,  epoch, mode="LSTM", model_name="",  model_dir = "model/cm"):
        
        test_loader     = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion   = nn.CrossEntropyLoss()
         # 在测试集上评估
         
        if mode == "INCEPTION":
            model = StockInceptionTime(input_dim, hidden_dim, num_layers, num_classes, dropout)
            model.load_state_dict(torch.load(model_name))

        if mode == "LSTM":
            model = StockLSTM(input_dim, hidden_dim, num_layers, num_classes, dropout)
            model.load_state_dict(torch.load(model_name))
        else:
            model = StockTransformer(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout)
            model.load_state_dict(torch.load(model_name))

        model.to(device)  # 将模型移动到指定的设备上
      
        test_loss,   total_accuracy, class_accuracies= CM.evaluate_with_cm(model, test_loader, criterion, device, ["DN", "NO", "UP"],  epoch, model_dir)


        print(f"測試集 殘差值: {test_loss:.4f},  測試集 準確值 {total_accuracy:.2%}")
        print(f"正確答案: {targets      [:20]}, 跌巿準確 : {class_accuracies[0]:.2%}, 橫盤準確 : {class_accuracies[1]:.2%}, 升巿準確 : {class_accuracies[2]:.2%}")  # 顯示前五個實際標籤
