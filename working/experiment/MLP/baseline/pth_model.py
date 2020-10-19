import os
import gc
import random
import math
import time
import datetime
import shutil

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import log_loss

import category_encoders as ce
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import common as com
import pth_modeling
import pth_preprocessing
import pth_model
from config import IO_CFG
from config import MODEL_CFG

log_folder = IO_CFG.input_root + '/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, 'pth_model.py')
###############################################################################################################

class TabularNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(cfg.num_features, cfg.hidden_size),
                          nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size, cfg.hidden_size),
                          nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size, cfg.target_cols),
                          )

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x