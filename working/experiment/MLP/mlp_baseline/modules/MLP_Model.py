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
    def __init__(self, cfg, num_features, target_cols):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(len(num_features))
        self.fc_block1 = FCBlock(len(num_features), cfg.hidden_size)
        self.fc_block2 = FCBlock(cfg.hidden_size, cfg.hidden_size)
        self.fc_moa = nn.Linear(cfg.hidden_size, len(target_cols))

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = cont_x
        x = self.bn0(x)
        
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        output = self.fc_moa(x)

        return output
