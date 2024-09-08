from torch import nn
import torch
from torch.nn import Module
import os
import pdb

class AFF_SmirkEmoRecognition(nn.Module):
    def __init__(self, input_size=358, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(AFF_SmirkEmoRecognition, self).__init__()
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
        #x = self.dropout2(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        expression = output[:, :8]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return valence, arousal, expression
        
    
class AFF_EmocaRecognition(nn.Module): 
    def __init__(self, input_size=156, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(AFF_EmocaRecognition, self).__init__()
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

    def forward(self, x, emoca_feature):  # residual connection #输入358 json file
        
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
        expression = output[:, :8]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return valence, arousal, expression
    

class RAF_SmirkRecognition(nn.Module):
    def __init__(self, input_size=358, hidden_size=2048, output_size=10, dropout_probs=[0.3, 0.3, 0.2]):
        super(RAF_SmirkRecognition, self).__init__()
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
        expression = output[:, :7]
        arousal = output[:, 8] 
        valence = output[:, 9] 
        return expression

        
class EmocaRecognition(nn.Module):
    def __init__(self, input_size=334, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(EmocaRecognition, self).__init__()
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
        expression = output[:, :7]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        return expression

