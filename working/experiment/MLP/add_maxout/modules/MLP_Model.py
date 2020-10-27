import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
import common as com
com.seed_everything(seed=42)

class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(FCBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.PReLU()
        
    def forward(self, input):
        x = input
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=CFG.dropout, training=self.training)
        x = self.act1(x)
        return x

class TabularNN(nn.Module):
    def __init__(self, cfg, num_features, target_cols, num_units = 5):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(len(num_features))
        self.fc_block1 = FCBlock(len(num_features), cfg.hidden_size)
        self.fc_block2 = FCBlock(cfg.hidden_size, cfg.hidden_size)
        self.fc_moa = nn.ModuleList([nn.Linear(cfg.hidden_size,len(target_cols)) for i in range (num_units)])
        
    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = cont_x
        x = self.bn0(x)
        
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        output = self.maxout(x, self.fc_moa)

        return output