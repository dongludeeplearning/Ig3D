from torch import nn
import torch
from networks import MixedFeatureNet
from torch.nn import Module
import os

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SmirkNet(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(SmirkNet, self).__init__()
        self.Smirk_Linear = nn.Linear(358, 512)

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

        self.fc = nn.Linear(512, num_class)
        
    def forward(self, x, smirk_feature):  # residual connection # 358 json file
        epsilon = 1e-10
        device = x.device
        smirk_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
        ]

        smirk_features = torch.cat(smirk_features_list, dim=1)  # torch.Size([256, 358])
        smirk_features_out = self.Smirk_Linear(smirk_features)  # torch.Size([256, 512])

        # Residual connection for the first block
        residual = smirk_features_out
        smirk_features_out = self.residual_block1(smirk_features_out)
        smirk_features_out += residual

        # Residual connection for the second block
        residual = smirk_features_out
        smirk_features_out = self.residual_block2(smirk_features_out)
        smirk_features_out += residual

        out = self.fc(smirk_features_out)  # torch.Size([256, 8])
        
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512,512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32): 
        super(CoordAtt, self).__init__()
      
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))    
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        
        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        
        identity = x 
        n,c,h,w = x.size()
        x_h = self.Linear_h(x) 
        x_w = self.Linear_w(x) 
        x_w = x_w.permute(0, 1, 3, 2) 
      
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2) 
        x_w = x_w.permute(0, 1, 3, 2)#torch.Size([256, 16, 1, 7])

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)#torch.Size([256, 512, 7, 7])
        x_w = x_w.expand(-1, -1, h, w)#torch.Size([256, 512, 7, 7])
     
        y = x_w * x_h #torch.Size([256, 512, 7, 7])
 
        return y
if __name__ == '__main__':
    smirk_feature = {
    "key1": [torch.randn(358) for _ in range(32)],
    "key2": [torch.randn(358) for _ in range(32)]
}
    model = SmirkNet()
    x = torch.randn(32, 100)  # Dummy input for x
    out = model(x, smirk_feature)