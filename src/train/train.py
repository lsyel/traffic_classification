from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import os
import sys
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))

sys.path.append(model_path)
from densenet import DenseNetBC100
from resnet import ResNet34

# ---- 新增1：数据预处理增强 ----
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset ='ustc2016'
train_dataset = ImageFolder(f"/root/wzhdesign/traffic_classification/dataset/{dataset}/final_data/train", transform=transform)
test_dataset = ImageFolder(f"/root/wzhdesign/traffic_classification/dataset/{dataset}/final_data/test", transform=transform)
# 打印类别标签对应关系
print("\n=== 类别标签对应关系 ===")
print(f"训练集类别数量: {len(train_dataset.classes)}")
print("标签索引 -> 类别名称:")
for i, class_name in enumerate(train_dataset.classes):
    print(f"{i} -> {class_name}")

# 确保测试集类别顺序相同
print("\n测试集类别顺序验证:")
print(f"测试集类别数量: {len(test_dataset.classes)}")
for i, class_name in enumerate(test_dataset.classes):
    print(f"{i} -> {class_name}")

# 检查训练集和测试集类别是否一致
if train_dataset.classes != test_dataset.classes:
    print("\n⚠️ 警告: 训练集和测试集的类别顺序不一致！")
    print("训练集类别:", train_dataset.classes)
    print("测试集类别:", test_dataset.classes)
else:
    print("\n✅ 训练集和测试集类别顺序一致")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 初始化模型
# model = DenseNetBC100(num_c=len(train_dataset.classes))
model = ResNet34(num_c=len(train_dataset.classes))
scaler = GradScaler()  # 混合精度缩放器


# 检查GPU可用性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型移至GPU
model = model.to(device)

optimizer = optim.AdamW(  # 改用AdamW
    model.parameters(),
    lr=0.001,
    weight_decay=0.01     # 增强正则化
)
criterion = nn.CrossEntropyLoss()

# ---- 修改4：动态学习率调度 ----
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5,
    patience=2,
    verbose=True
)



def train():
        # ---- 修改5：增强的训练循环 ----
    best_val_acc = 0.0
    patience_counter = 0

    # 早停参数
    best_val_acc = 0.0
    patience = 10
    no_improve_epochs = 0
    num_epochs = 10

    from tqdm import tqdm
    import time

    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # 训练阶段（混合精度）
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():  # 混合精度前向
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()  # 缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # 验证阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = 100 * correct / len(test_dataset)
        scheduler.step(val_acc)  # 动态调整学习率
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{dataset}.pth")
            print(f"🏆 最佳模型保存，准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 早停触发")
                break
        
        # 打印统计信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Loss: {epoch_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    print("Training complete!")
    model.load_state_dict(torch.load(f"best_model_{dataset}.pth"))
    model.eval()
def eval():
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from tqdm import tqdm

    # 混淆矩阵可视化
    def plot_confusion_matrix(labels, preds, classes):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("混淆矩阵")
        plt.savefig("confusion_matrix.png")  # 保存图片
        plt.close()
    # 加载最佳模型
    model.load_state_dict(torch.load(f"best_model_{dataset}.pth"))
    model.eval()

    # 初始化存储变量
    all_labels = []
    all_preds = []

    # 测试集评估
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试中"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算指标
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🏁 最终测试准确率: {accuracy:.2f}%")

    # 分类报告（精确率/召回率/F1）
    print("\n📊 分类报告:")
    print(classification_report(
        all_labels, all_preds, 
        target_names=test_dataset.classes,  # 替换为你的类别名称列表
        digits=4
    ))


    #混淆矩阵输出
    print("\n📊 混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))
    #矩阵可视化
    plot_confusion_matrix(all_labels, all_preds, test_dataset.classes)
if __name__ == '__main__':
    train()
    eval()