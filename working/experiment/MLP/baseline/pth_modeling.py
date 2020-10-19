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
####################################################################################################################

def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    
    losses = AverageMeter()

    model.train()

    for step, (cont_x, cate_x, y) in enumerate(train_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cate_x)
        
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        if MODEL_CFG.gradient_accumulation_steps > 1:
            loss = loss / MODEL_CFG.gradient_accumulation_steps

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MODEL_CFG.max_grad_norm)

        if (step + 1) % MODEL_CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        
    return losses.avg


def validate_fn(valid_loader, model, device):
    
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cate_x, y) in enumerate(valid_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)
            
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        val_preds.append(pred.sigmoid().detach().cpu().numpy())

        if MODEL_CFG.gradient_accumulation_steps > 1:
            loss = loss / MODEL_CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)
        
    return losses.avg, val_preds


def inference_fn(test_loader, model, device):

    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):

        cont_x,  cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

