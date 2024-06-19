import torch
import torch.nn as nn


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


class InceptionModule(nn.Module):
    def __init__(self, in_channels, filters):
        super(InceptionModule, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, filters, kernel_size=1, padding=0, bias=False)
        self.mp_bottleneck = nn.Conv1d(in_channels, filters, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(filters, filters, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(filters, filters, kernel_size=7, padding=3, bias=False)
        self.mp = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm1d(filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        conv3_output = self.conv3(bottleneck_output)
        conv5_output = self.conv5(bottleneck_output)
        conv7_output = self.conv7(bottleneck_output)
        mp_output = self.mp(x)
        mp_bottleneck_output = self.mp_bottleneck(mp_output)

        output = torch.cat([conv3_output, conv5_output, conv7_output, mp_bottleneck_output], dim=1)
        output = self.bn(output)
        output = self.relu(output)

        return output


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutLayer, self).__init__()
        self.conv       = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn         = nn.BatchNorm1d(out_channels)
        self.relu       = nn.ReLU()

    def forward(self, x, shortcut):
        shortcut_output = self.conv(shortcut)
        shortcut_output = self.bn(shortcut_output)

        output = shortcut_output + x
        output = self.relu(output)

        return output


class StockInceptionTime(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1, l1_lambda=0.0, l2_lambda=0.0):
        super(StockInceptionTime, self).__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.inception_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.inception_layers.append(InceptionModule(input_dim, hidden_dim))
            else:
                self.inception_layers.append(InceptionModule(hidden_dim * 4, hidden_dim))

            if i % 3 == 2:
                if i == 2:
                    self.shortcut_layers.append(ShortcutLayer(input_dim, hidden_dim * 4))
                else:
                    self.shortcut_layers.append(ShortcutLayer(hidden_dim * 4, hidden_dim * 4))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        input_residual = x

        for i in range(self.num_layers):
            x = self.inception_layers[i](x)
            x = self.dropout_layer(x)

            if i % 3 == 2:
                shortcut_index = i // 3
                x = self.shortcut_layers[shortcut_index](x, input_residual)
                input_residual = x

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # 计算 L1 和 L2 正则化项
        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l1_reg = l1_reg + torch.norm(param, 1)
                l2_reg = l2_reg + torch.norm(param, 2)**2

        return x, l1_reg, l2_reg,  self.l1_lambda, self.l2_lambda