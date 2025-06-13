import os
import shutil

def merge_with_source_prefix(src1, src2, dest, conflict_mode='rename'):
    """
    合并两个文件夹，生成带源文件夹名前缀的新文件名
    
    参数：
        src1 (str): 第一个源文件夹路径（如 `/data/folder1`）
        src2 (str): 第二个源文件夹路径
        dest (str): 目标文件夹路径
        conflict_mode (str): 冲突处理方式
            'rename'    - 自动重命名 (默认)
            'overwrite' - 覆盖
            'skip'      - 跳过
    """
    os.makedirs(dest, exist_ok=True)
    
    for src in [src1, src2]:
        # 获取源文件夹基名（如 "folder1"）
        src_name = os.path.basename(os.path.normpath(src))
        
        for root, dirs, files in os.walk(src):
            #一半文件
            for file in files[::2]:
                src_path = os.path.join(root, file)
                
                # 生成新文件名：源文件夹名_子目录结构_文件名
                rel_path = os.path.relpath(src_path, src)
                new_name = f"{src_name}_{rel_path.replace(os.sep, '_')}"
                dest_path = os.path.join(dest, new_name)
                
                # 处理重名冲突
                if os.path.exists(dest_path):
                    if conflict_mode == 'skip':
                        print(f"跳过: {dest_path} (已存在)")
                        continue
                    elif conflict_mode == 'overwrite':
                        print(f"覆盖: {dest_path}")
                    elif conflict_mode == 'rename':
                        base, ext = os.path.splitext(new_name)
                        count = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(dest, f"{base} ({count}){ext}")
                            count += 1
                        print(f"重命名: {new_name} -> {os.path.basename(dest_path)}")
                
                # 复制文件
                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"成功: {src_path} -> {dest_path}")
                except Exception as e:
                    print(f"错误: 无法复制 {src_path} -> {dest_path}\n原因: {str(e)}")

if __name__ == "__main__":
    merge_with_source_prefix(
        src1="dataset/ustc2016/final_data/test/Gmail",
        src2="dataset/ustc2016/final_data/test/Outlook",
        dest="dataset/ustc2016/final_data/test/Gmail_Outlook",
        conflict_mode='rename'
    )