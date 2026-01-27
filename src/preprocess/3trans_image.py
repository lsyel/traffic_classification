import time
from scapy.all import rdpcap
from PIL import Image
import numpy as np
import os
from scapy.all import rdpcap, IP, TCP, UDP, Raw
import entropy

def shannon_entropy(byte_array):
    """手动实现香农熵计算"""
    value, counts = np.unique(byte_array, return_counts=True)
    prob = counts / len(byte_array)
    return -np.sum(prob * np.log2(prob))  # 单位：bits

def flow_to_multichannel_image(flow_pcap_path, img_size=(32, 32)):
    packets = rdpcap(flow_pcap_path)
    
    if len(packets) == 0:
        return Image.new("RGB", img_size, color=(0, 0, 0))
    
    # 通道1: 包长度（全局归一化）
    pkt_lengths = np.array([len(pkt) for pkt in packets], dtype=np.float32)
    pkt_lengths = np.interp(pkt_lengths, [0, 1500], [0, 255]).astype(np.uint8)
    
    # 通道2: 协议+端口
    proto_features = []
    for pkt in packets:
        if pkt.haslayer(IP):
            proto = pkt[IP].proto
            # 提取源/目的端口（兼容TCP/UDP）
            sport = pkt.sport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else 0
            dport = pkt.dport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else 0
            proto_features.append([
                proto,
                sport // 256, sport % 256,  # 拆分为高8位和低8位
                dport // 256, dport % 256
            ])
        else:
            proto_features.append([0] * 5)
    proto_features = np.array(proto_features, dtype=np.uint8)
    
    # 通道3: 负载统计（均值+熵）
    payload_stats = []
    for pkt in packets:
        if pkt.haslayer(Raw):
            payload = bytes(pkt[Raw])
            if payload:
                byte_array = np.frombuffer(payload, dtype=np.uint8)
                mean = np.mean(byte_array)
                ent = shannon_entropy(byte_array) * 10  # 放大以便可视化
                payload_stats.append([int(mean), int(ent)])
            else:
                payload_stats.append([0, 0])
        else:
            payload_stats.append([0, 0])
    payload_stats = np.array(payload_stats, dtype=np.uint8)
    
    # 合并通道（时间步, 8通道）
    combined = np.concatenate([
        pkt_lengths[:, None],  # 添加新轴 (T,1)
        proto_features,         # (T,5)
        payload_stats           # (T,2)
    ], axis=1)
    
    # 调整时间步长
    target_timesteps = img_size[0]
    if combined.shape[0] < target_timesteps:
        combined = np.pad(combined, ((0, target_timesteps - combined.shape[0]), (0, 0)), mode="edge")
    else:
        combined = combined[:target_timesteps, :]
    
    # 转换为图像
    img = Image.fromarray(combined).resize(
        (img_size[1], target_timesteps),
        resample=Image.NEAREST
    )
    return img.convert("RGB")


dataset_name = 'ustc2016'
# 处理所有流
start_time = time.time()
image_count = 0
for split in ["train", "test"]:
    split_dir = os.path.join("/root/wzhdesign/traffic_classification/dataset/{}/final_data".format(dataset_name), split)
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        for flow_file in os.listdir(label_dir):
            flow_path = os.path.join(label_dir, flow_file)
            # image = flow_to_image(flow_path)
            image = flow_to_multichannel_image(flow_path)
            image.save(os.path.join(label_dir, f"{label}_{os.path.splitext(flow_file)[0]}.png"))
            os.remove(flow_path)  # 删除原始 pcap
            image_count += 1
end_time = time.time()
avg_time = (end_time - start_time) / image_count
print("转换完成，耗时: {} 秒".format(end_time - start_time))
print("转换完成，共生成 {} 张图像".format(image_count))
print("平均每张图像耗时: {} 秒".format(avg_time))


