from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import os

from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import os

def split_pcap_by_flows(input_pcap, output_dir, label, max_flows_per_class=2000):
    """将原始 pcap 按流分割为多个子 pcap，每个类别最多保留指定数量的流"""
    packets = rdpcap(input_pcap)
    flows = defaultdict(list)
    
    # 按五元组分组流
    for pkt in packets:
        if pkt.haslayer("IP"):
            src = pkt["IP"].src
            dst = pkt["IP"].dst
            proto = pkt["IP"].proto
            sport = pkt.sport if pkt.haslayer("TCP") else pkt.sport if pkt.haslayer("UDP") else 0
            dport = pkt.dport if pkt.haslayer("TCP") else pkt.dport if pkt.haslayer("UDP") else 0
            # 统一源和目的顺序
            if src > dst:
                src, dst = dst, src
                sport, dport = dport, sport
            flow_key = (src, dst, proto, sport, dport)
            flows[flow_key].append(pkt)
    
    # 确保输出目录存在
    output_label_dir = os.path.join(output_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取当前已存在的流文件数量
    existing_flows = len([f for f in os.listdir(output_label_dir) if f.startswith("flow_") and f.endswith(".pcap")])
    remaining_slots = max(0, max_flows_per_class - existing_flows)
    
    if remaining_slots == 0:
        print(f"⚠️ 类别 {label} 已达到 {max_flows_per_class} 流上限，跳过处理")
        return
    
    # 保存新流，直到达到上限
    saved_count = 0
    for i, (flow_key, flow_packets) in enumerate(flows.items()):
        if saved_count >= remaining_slots:
            break
        output_path = os.path.join(output_label_dir, f"flow_{existing_flows + saved_count}.pcap")
        wrpcap(output_path, flow_packets)
        saved_count += 1
    
    print(f"✅ 类别 {label} 新增 {saved_count} 个流，总流数 {existing_flows + saved_count}/{max_flows_per_class}")
dataset_name = 'ustc2016_all'
# 处理所有 pcap 文件
input_dir = "/root/wzhdesign/traffic_classification/dataset/{}/raw_data".format(dataset_name)
output_dir = "/root/wzhdesign/traffic_classification/dataset/{}/split_flows".format(dataset_name)

for filename in os.listdir(input_dir):
    if filename.endswith(".pcap"):
        label = os.path.splitext(filename)[0]  # 从文件名获取类别
        split_pcap_by_flows(
            os.path.join(input_dir, filename),
            output_dir,
            label
        )