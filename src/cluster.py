import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')
# === 加入这几行 ===
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

def extract_features(dataset, batch_size=32, device='cpu'):
    """
    使用预训练的ResNet模型提取图像特征
    """
    # 加载预训练的ResNet模型
    model = models.resnet18(pretrained=True)
    # 移除最后的全连接层，只保留特征提取部分
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # 提取特征
            output = model(images)
            # 展平特征
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.extend(targets.numpy())
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    return features, labels

def perform_clustering(features, n_clusters=5):
    """
    使用K-means算法进行聚类
    """
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    return cluster_labels, kmeans, scaler

def visualize_results(features, cluster_labels, true_labels, dataset, n_images_per_cluster=5):
    """
    可视化聚类结果
    """
    # 使用t-SNE进行降维以便可视化
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 聚类结果散点图
    scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, 
                                cmap='viridis', alpha=0.7)
    # axes[0, 0].set_title('聚类结果 (t-SNE 可视化)')
    axes[0, 0].set_xlabel('t-SNE 特征 1')
    axes[0, 0].set_ylabel('t-SNE 特征 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. 真实标签散点图（如果有的话）
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        scatter_true = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, 
                                         cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('真实标签 (t-SNE 可视化)')
        axes[0, 1].set_xlabel('t-SNE 特征 1')
        axes[0, 1].set_ylabel('t-SNE 特征 2')
        plt.colorbar(scatter_true, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, '无真实标签信息', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('真实标签')
    
    # 3. 每个聚类的示例图像
    unique_clusters = np.unique(cluster_labels)
    
    # 创建一个子图网格来显示示例图像
    grid_size = (len(unique_clusters), n_images_per_cluster)
    
    # 调整子图布局
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig2 = plt.figure(figsize=(15, 3*len(unique_clusters)))
    grid = ImageGrid(fig2, 111, nrows_ncols=grid_size, axes_pad=0.1)
    
    # 获取每个聚类的图像索引
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # 随机选择n_images_per_cluster个图像
        if len(cluster_indices) > n_images_per_cluster:
            selected_indices = np.random.choice(cluster_indices, n_images_per_cluster, replace=False)
        else:
            selected_indices = cluster_indices
        
        # 显示图像
        for j, idx in enumerate(selected_indices):
            ax = grid[i * n_images_per_cluster + j]
            image_path, _ = dataset.samples[idx]
            image = Image.open(image_path)
            ax.imshow(image)
            ax.set_title(f'聚类 {cluster_id}')
            ax.axis('off')
    
    # 4. 聚类大小柱状图
    cluster_sizes = [np.sum(cluster_labels == i) for i in unique_clusters]
    axes[1, 0].bar(unique_clusters, cluster_sizes)
    axes[1, 0].set_title('每个聚类的图像数量')
    axes[1, 0].set_xlabel('聚类 ID')
    axes[1, 0].set_ylabel('图像数量')
    
    # 5. 特征重要性（使用PCA）
    pca = PCA(n_components=min(10, features.shape[1]))
    pca.fit(features)
    axes[1, 1].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    axes[1, 1].set_title('PCA 解释方差比')
    axes[1, 1].set_xlabel('主成分')
    axes[1, 1].set_ylabel('解释方差比例')
    
    plt.tight_layout()
    plt.show()
    
    return fig, fig2

def main():
    # 设置数据集路径
    data_path = "/root/wzhdesign/traffic_classification/dataset/cluster"  # 请替换为你的数据集路径
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 路径 '{data_path}' 不存在!")
        print("请将 data_path 变量设置为你的图像数据集路径")
        return
    
    # 加载图像数据集
    print("正在加载图像数据集...")
    dataset = ImageFolder(data_path)
    print(f"数据集包含 {len(dataset)} 张图像, {len(dataset.classes)} 个类别")
    print(f"类别: {dataset.classes}")
    
    # 提取特征
    print("正在提取图像特征...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    features, true_labels = extract_features(dataset, device=device)
    print(f"特征维度: {features.shape}")
    
    # 确定聚类数量（如果类别数已知，可以使用类别数；否则使用经验法则）
    if len(dataset.classes) > 1:
        n_clusters = len(dataset.classes)
    else:
        # 使用肘部法则或轮廓系数确定最佳聚类数
        n_clusters = min(4, max(2, len(dataset) // 50))  # 简单的启发式方法
    
    print(f"使用 {n_clusters} 个聚类")
    
    # 执行聚类
    print("正在进行聚类分析...")
    cluster_labels, kmeans, scaler = perform_clustering(features, n_clusters=n_clusters)
    
    # 可视化结果
    print("正在生成可视化结果...")
    fig, fig2 = visualize_results(features, cluster_labels, true_labels, dataset)
    

    # 保存可视化图像
    fig.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
    # fig2.savefig('cluster_examples.png', dpi=300, bbox_inches='tight')
    print("结果已保存!")

    # 打印聚类统计信息
    print("\n聚类统计信息:")
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        print(f"聚类 {i}: {cluster_size} 张图像 ({cluster_size/len(dataset)*100:.1f}%)")
        
        # 如果真实标签可用，显示每个聚类中的类别分布
        if true_labels is not None and len(np.unique(true_labels)) > 1:
            cluster_classes = true_labels[cluster_labels == i]
            unique, counts = np.unique(cluster_classes, return_counts=True)
            class_dist = ", ".join([f"{dataset.classes[cls]}: {cnt}" for cls, cnt in zip(unique, counts)])
            print(f"  类别分布: {class_dist}")

if __name__ == "__main__":
    main()