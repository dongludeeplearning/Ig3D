from torch import nn
import torch
from torch.nn import Module
import os

class EmocaEmoRecognition_F(nn.Module):
    def __init__(self, input_size=334, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(EmocaEmoRecognition_F, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_probs[0])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_probs[1])
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_probs[2])
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, emoca_feature):  
        
        device = x.device
        emoca_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in emoca_feature[key]], dim=1) for key in emoca_feature
        ]

        emoca_features = torch.cat(emoca_features_list, dim=1).float()  # 确保数据类型为 float32
        x = self.leaky_relu(self.bn1(self.fc1(emoca_features)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        expression = output[:, :8]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return valence, arousal, expression
    
    
class EmocaEmoRecognition_S(nn.Module):
    def __init__(self, input_size=156, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(EmocaEmoRecognition_S, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_probs[0])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_probs[1])
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_probs[2])
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, emoca_feature):  
        
        device = x.device
        emoca_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in emoca_feature[key]], dim=1) for key in emoca_feature
        ]

        emoca_features = torch.cat(emoca_features_list, dim=1).float()  # 确保数据类型为 float32
     
        x = self.leaky_relu(self.bn1(self.fc1(emoca_features)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        expression = output[:, :8]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return valence, arousal, expression
    
    
class SmirkEmoRecognition_S(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(SmirkEmoRecognition_S, self).__init__()
        self.Smirk_Linear = nn.Linear(353, 512)
        self.num_class = num_class

        self.residual_block1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )

        self.residual_block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )

        self.fc = nn.Linear(512, num_class + 2)
        
    def forward(self, x, smirk_feature):  
        epsilon = 1e-10
        device = x.device
        smirk_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
        ]

        smirk_features = torch.cat(smirk_features_list, dim=1).float()  
        smirk_features_out = self.Smirk_Linear(smirk_features)  
        residual = smirk_features_out
        smirk_features_out = self.residual_block1(smirk_features_out)
        smirk_features_out += residual
        residual = smirk_features_out
        smirk_features_out = self.residual_block2(smirk_features_out)
        smirk_features_out += residual
        out = self.fc(smirk_features_out)  
        out_cls = out[:, :self.num_class]  
        out_val = out[:, self.num_class]   
        out_aro = out[:, self.num_class + 1]  
        
        return out_cls, out_val, out_aro
    
class SmirkEmoRecognition_F(nn.Module):
    def __init__(self, input_size=358, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
            super(SmirkEmoRecognition_F, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(dropout_probs[0])
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.dropout2 = nn.Dropout(dropout_probs[1])
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
            self.dropout3 = nn.Dropout(dropout_probs[2])
            self.fc4 = nn.Linear(hidden_size, hidden_size)
            self.bn4 = nn.BatchNorm1d(hidden_size)
            self.output = nn.Linear(hidden_size, output_size)
            self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, smirk_feature):  # residual connection #输入358 json file
            device = x.device
            smirk_features_list = [
                torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
            ]

            smirk_features = torch.cat(smirk_features_list, dim=1).float()  # 确保数据类型为 float32
            x = self.leaky_relu(self.bn1(self.fc1(smirk_features)))
            x = self.dropout1(x)
            x = self.leaky_relu(self.bn2(self.fc2(x)))
        
            x = self.leaky_relu(self.bn3(self.fc3(x)))
            x = self.leaky_relu(self.bn4(self.fc4(x)))
            output = self.output(x)
            expression = output[:, :8]  # [64,8]
            arousal = output[:, 8]  # [64]
            valence = output[:, 9]  # [64]
            return valence, arousal, expression

class RAFDB_Smirk_Recognition(nn.Module):
    def __init__(self, input_size=353, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(RAFDB_Smirk_Recognition, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_probs[0])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_probs[1])
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_probs[2])
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, x, emoca_feature):  
        device = x.device
        emoca_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in emoca_feature[key]], dim=1) for key in emoca_feature
        ]
        emoca_features = torch.cat(emoca_features_list, dim=1).float()
        x = self.leaky_relu(self.bn1(self.fc1(emoca_features)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        expression = output[:, :7]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return expression
    

class RAFDB_Emoca_Recognition(nn.Module):
    def __init__(self, input_size=156, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(RAFDB_Emoca_Recognition, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_probs[0])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_probs[1])
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_probs[2])
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, emoca_feature):  
        
        device = x.device
        emoca_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in emoca_feature[key]], dim=1) for key in emoca_feature
        ]

        emoca_features = torch.cat(emoca_features_list, dim=1).float()  # 确保数据类型为 float32
        x = self.leaky_relu(self.bn1(self.fc1(emoca_features)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        #x = self.dropout2(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        expression = output[:, :7]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return expression
