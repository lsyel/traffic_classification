import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os
from scapy.all import rdpcap, wrpcap    
from collections import defaultdict
# 划分参数
test_size = 0.2
random_state = 42
dataset_name = 'ustc2016_all'
if dataset_name == 'ustc2016':
    test_size =0.2

# 处理所有 pcap 文件
output_dir = "dataset/{}/split_flows".format(dataset_name)
# 创建目标目录
os.makedirs("/root/wzhdesign/traffic_classification/dataset/{}/final_data/train".format(dataset_name), exist_ok=True)
os.makedirs("/root/wzhdesign/traffic_classification/dataset/{}/final_data/test".format(dataset_name), exist_ok=True)

for label in os.listdir(output_dir):
    label_dir = os.path.join(output_dir, label)
    flow_files = [f for f in os.listdir(label_dir) if f.endswith(".pcap")]
    
    # 划分训练/测试集
    train_files, test_files = train_test_split(
        flow_files,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # 复制训练集
    train_label_dir = os.path.join("dataset/{}/final_data/train".format(dataset_name), label)
    os.makedirs(train_label_dir, exist_ok=True)
    for f in train_files:
        src = os.path.join(label_dir, f)
        dst = os.path.join(train_label_dir, f)
        shutil.copy(src, dst)
    
    # 复制测试集
    test_label_dir = os.path.join("dataset/{}/final_data/test".format(dataset_name), label)
    os.makedirs(test_label_dir, exist_ok=True)
    for f in test_files:
        src = os.path.join(label_dir, f)
        dst = os.path.join(test_label_dir, f)
        shutil.copy(src, dst)