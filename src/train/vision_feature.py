
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
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
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
sys.path.append(model_path)
from resnet import ResNet34
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 1. 数据加载设置
def load_data(data_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    return dataset, loader

# 2. 特征提取函数（修复版本）
def extract_features(model, dataloader, class_idx, device):
    features = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            mask = (labels == class_idx)
            if torch.any(mask):
                target_images = images[mask]
                _, features_batch = model.penultimate_forward(target_images)
                pooled = torch.mean(features_batch, dim=(2, 3))
                features.append(pooled.cpu().numpy())
    
    return np.concatenate(features, axis=0) if features else np.array([])

# 3. 多聚类验证可视化
def visualize_multiple_clusters(features, class_name):
    if len(features) == 0:
        print(f"错误：{class_name}类别没有特征数据")
        return None
    
    # 随机采样300个点以便可视化更清晰（当数据量大时）
    if len(features) > 1500:
        np.random.seed(42)
        indices = np.random.choice(len(features), 1500, replace=False)
        features = features[indices]
    
    plt.figure(figsize=(18, 16))
    
    # 主成分分析投影
    plt.subplot(2, 2, 1)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)
    
    # 尝试聚类（假设3-5个集群）
    n_clusters = min(7, max(2, len(features)//100))  # 动态确定聚类数量
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # 计算轮廓系数
    try:
        silhouette = silhouette_score(features, cluster_labels)
    except ValueError:
        silhouette = -1.0  # 当聚类数不合适时
    
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], 
                          c=cluster_labels, cmap='tab10', s=25, alpha=0.8)
    plt.title(f'PCA Projection Clustering ({n_clusters} Subclasses)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(*scatter.legend_elements(), title="Subclasses")
    
    # t-SNE投影
    plt.subplot(2, 2, 2)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                c=cluster_labels, cmap='tab10', s=25, alpha=0.8)
    plt.title('t-SNE Projection Clustering', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)    
    
    # 层次聚类树状图
    plt.subplot(2, 2, 3)
    linked = linkage(features, 'ward')
    dendrogram(linked,
               orientation='top',
               truncate_mode='lastp',
               p=12,  # 显示最后12次合并
               show_leaf_counts=True,
               leaf_rotation=90.,
               leaf_font_size=12.,
               show_contracted=True)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Cluster Distance', fontsize=14)
    plt.axhline(y=linked[-4, 2], c='r', linestyle='--')
    
    # 子类特征比较热力图
    plt.subplot(2, 2, 4)
    cluster_means = []
    for i in range(n_clusters):
        cluster_features = features[cluster_labels == i]
        cluster_means.append(np.mean(cluster_features, axis=0))
    
    # 只显示最有区分度的前50个特征维度
    variances = np.var(cluster_means, axis=0)
    top_features = np.argsort(variances)[-50:][::-1]
    
    # 创建热力图数据
    heatmap_data = []
    for i in range(n_clusters):
        heatmap_data.append(cluster_means[i][top_features])
    
    sns.heatmap(np.array(heatmap_data).T, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Subclass Feature Differences', fontsize=16)
    plt.xlabel('Subclass ID', fontsize=14)
    plt.ylabel('Feature Dimensions', fontsize=14)
    plt.yticks([])  # 隐藏y轴刻度（特征维度太多）
    
    plt.tight_layout()
    plt.suptitle(f'', fontsize=20, y=0.98)
    plt.savefig(f'{class_name}_clustering_visualization.png', dpi=300)
    plt.show()
    
    # 打印关键统计信息
    print(f"\n{'-'*40}")
    print(f"{class_name}类聚类验证结果")
    print(f"分析样本数: {len(features)}")
    print(f"最优化聚类数: {n_clusters}")
    print(f"轮廓系数: {silhouette:.4f} (越高越好)")
    print(f"树状图分支数: {np.sum(linked[-4:, 2] > linked[-5, 2])}个显著分离群")
    
    # 判断是否存在多个子类
    if n_clusters > 1 and silhouette > 0.5:
        print("\n✅ 明确证据表明存在多个子类结构")
        print("子类间特征差异: 显著（热力图显示明显颜色变化）")
    elif n_clusters > 1:
        print("\n🟡 存在子类结构但分离不明显")
        print("建议: 尝试非线性聚类方法如DBSCAN")
    else:
        print("\n❌ 未发现明显子类结构")
        print("所有样本在特征空间中紧密聚集")
    
    print(f"{'-'*40}\n")
    
    return cluster_labels

# 主程序
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = "dataset/ustc2016/final_data/train"  # 请替换为您的实际路径
    dataset, train_loader = load_data(data_path)
    
    # 加载模型 (从您的代码中导入ResNet34和ResNet类)
    
    num_classes = len(dataset.classes)
    model = ResNet34(num_classes).to(device)
    model.load_state_dict(torch.load('best_model_ustc2016.pth', map_location=device))
    model.eval()
    
    # # 选择BitTorrent类
    # target_class_name = "BitTorrent"
    # target_class_idx = dataset.class_to_idx[target_class_name]
    for target_class_name in dataset.classes:
        target_class_idx = dataset.class_to_idx[target_class_name]
        
        print(f"提取 {target_class_name} 类别的特征...")
        class_features = extract_features(model, train_loader, target_class_idx, device)
        
        if len(class_features) == 0:
            print(f"错误：未找到 {target_class_name} 类别的样本")
        else:
            print(f"成功提取 {len(class_features)} 个 {target_class_name} 样本特征")
            # 执行多聚类验证
            cluster_labels = visualize_multiple_clusters(class_features, target_class_name)