# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:11:41 2020

@author: ALW
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as ml
from torch import optim
from torch.autograd import Variable
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
#import loss
import random
import pandas as pd
import Muti_Modal as MM
from torch.utils.data import DataLoader, Dataset
from dataread import DatasetSplit, DatasetSplit_test,DatasetSplit_g
import warnings
warnings.filterwarnings("ignore")
SEED = 42  
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



batch_size = 900
batch_size1 = 900
lr = 1e-3
os.environ['CUDA_VISIBLE_DEVICES'] = '2'   ####gpu选择

dim = 40

net = MM.Mut_Modal(3)#.TransFormer(2)#ml.SKnet(1,2)
#net = torch.load("/data/nfs_rt16/luyuan/code/interspeech_classifition/ResNet_0207_augment/7_my_model.pth")#.module
#print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ###如果没有gpu选择cpu

if torch.cuda.device_count() > 1:
  net = nn.DataParallel(net)  ####gpu并行训练
# Assuming that we are on a CUDA machine, this should print a CUDA device:
net.to(device)  ####网络采用gpu
print(device)
net=net.double()  

criterion = torch.nn.CrossEntropyLoss()  #####CE准则loss
#criterion = torch.nn.SmoothL1Loss()
#criterion = loss.FocalLoss(5)
optimizer = optim.Adam(net.parameters(), lr=lr)  ########优化器


#data = pd.read_csv('/disk1/xly/un-planned_reoperation/data/data_lite.csv',engine='python',encoding='gbk')
# print(data)

def evaluate_model(net, loader, device, branch="all"):
    net.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data1, data2, data3, y in loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)[0]
            y = y.to(device).long()

            out_all, out_series, out_text = net(data1, data2, data3)

            if branch == "all":
                out = out_all
            elif branch == "series":
                out = out_series
            else:
                out = out_text

            pred = torch.argmax(out, dim=1)
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return {
        "acc": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro")
    }


data = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/xh_all_3.txt")
data1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/xh_all_text_CB.txt")

data_1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/cangzhou_3.txt")
data1_1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/cangzhou_text_CB.txt")

data_2 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/last_3.txt")
data1_2 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/last_text_CB.txt")

data = np.concatenate((data, data_1, data_2), axis=0)
data1 = np.concatenate((data1, data1_1, data1_2), axis=0)

data_test_ex = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/data_Lite_externel_3.txt")
data_test_ex1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/data_Lite_externel_text_CB.txt")

data_test_ex_V1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/5gezhongxin_3.txt")
data_test_ex1_V1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_12_15/5gezhongxin_text_CB.txt")

data = np.concatenate((data, data1), axis=1)
data_test_ex = np.concatenate((data_test_ex, data_test_ex1), axis=1)
data_test_ex_V1 = np.concatenate((data_test_ex_V1, data_test_ex1_V1), axis=1)
print(data.shape,data1.shape)
# 假设最后一列是目标变量
X = data[:, 2:]
y = data[:, 1]

X_ex = data_test_ex[:, 2:]
y_ex = data_test_ex[:, 1]

X_ex_V1 = data_test_ex_V1[:, 2:]
y_ex_V1 = data_test_ex_V1[:, 1]
print(X.shape,y.shape,X_ex.shape,y_ex.shape,X_ex_V1.shape,y_ex_V1.shape)


# X_ex = X_ex[:,:2324]
# X_ex_V1 = X_ex_V1[:,:2324]
# X = X[:,:2324]

print(X.shape,X_ex.shape,X_ex_V1.shape)
print(len(X),len(X_ex),len(X_ex_V1))

from sklearn.model_selection import KFold
from tqdm import tqdm

KFOLDS = 5
kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
num_epochs = 1000  # 根据你的训练轮数改
branches = ["all", "series", "text"]
splits = ["internal", "external", "external_v1"]

# cv_results[branch][split][metric] -> list of 5 folds
cv_results = {
    b: {s: {"acc": [], "recall": [], "f1": []} for s in splits} for b in branches
}

for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
    print(f"\n==============================")
    print(f"🔥 Fold {fold} / {KFOLDS}")
    print(f"==============================")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    trainloader = DataLoader(DatasetSplit_g(X_train, y_train), batch_size=512, shuffle=True, num_workers=8)
    testloader = DataLoader(DatasetSplit_g(X_test, y_test), batch_size=len(X_test), shuffle=False)
    testloader_ex = DataLoader(DatasetSplit_g(X_ex, y_ex), batch_size=len(X_ex), shuffle=False)
    testloader_ex_V1 = DataLoader(DatasetSplit_g(X_ex_V1, y_ex_V1), batch_size=len(X_ex_V1), shuffle=False)

    # ===== 初始化模型 =====
    net = MM.Mut_Modal(3).to(device).double()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.9)

    # ===== 当前 fold 的 best =====
    Results = {
        b: {s: {"acc": 0.0, "recall": 0.0, "f1": 0.0, "epoch": -1} for s in splits} for b in branches
    }

    # =====================
    #   Epoch loop with tqdm
    # =====================
    for epoch in tqdm(range(num_epochs), desc=f"Fold {fold} Epochs", ncols=100):
        net.train()
        for data1, data2, data3, labels in tqdm(trainloader, desc=f"Fold {fold} Training Batches", ncols=100, leave=False):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)[0]
            labels = labels.to(device).long()

            out_all, out_series, out_text = net(data1, data2, data3)
            loss = (
                0.6 * criterion(out_all, labels) +
                0.2 * criterion(out_series, labels) +
                0.2 * criterion(out_text, labels)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===== validation =====
        loaders = {
            "internal": testloader,
            "external": testloader_ex,
            "external_v1": testloader_ex_V1
        }

        for split, loader in loaders.items():
            for b in branches:
                res = evaluate_model(net, loader, device, branch=b)
                for metric in ["acc", "recall", "f1"]:
                    if res[metric] > Results[b][split][metric]:
                        Results[b][split][metric] = res[metric]
                        Results[b][split]["epoch"] = epoch

        scheduler.step()

        # ===== 每100轮打印当前最优结果 =====
        if (epoch + 1) % 1000 == 0 or epoch == num_epochs - 1:
            print(f"\nFold {fold} Epoch {epoch+1} Current Best Results:")
            for b in branches:
                for s in splits:
                    print(
                        f"[{b.upper()} | {s}] "
                        f"ACC={Results[b][s]['acc']:.4f}, "
                        f"Recall={Results[b][s]['recall']:.4f}, "
                        f"F1={Results[b][s]['f1']:.4f}"
                    )
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")

    # ===== fold 结束：存入五折统计 =====
    print(f"\n📌 Fold {fold} RESULTS")
    for b in branches:
        for s in splits:
            for m in ["acc", "recall", "f1"]:
                cv_results[b][s][m].append(Results[b][s][m])
            print(
                f"[{b.upper()} | {s}] "
                f"ACC={Results[b][s]['acc']:.4f}, "
                f"Recall={Results[b][s]['recall']:.4f}, "
                f"F1={Results[b][s]['f1']:.4f}"
            )

print("\n" + "="*70)
print("🏆 5-FOLD BEST RESULTS (MEAN ± STD)")
print("="*70)
for b in branches:
    print(f"\n🔹 Branch: {b.upper()}")
    for s in splits:
        accs = np.array(cv_results[b][s]["acc"])
        recalls = np.array(cv_results[b][s]["recall"])
        f1s = np.array(cv_results[b][s]["f1"])
        print(
            f"  [{s}] "
            f"ACC={accs.mean():.4f}±{accs.std():.4f}, "
            f"Recall={recalls.mean():.4f}±{recalls.std():.4f}, "
            f"F1={f1s.mean():.4f}±{f1s.std():.4f}"
        )
