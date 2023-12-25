import os
import numpy as np
import scipy.io

# 定义数据集路径
data_path = r"C:\Users\pym66\Documents\文献\writing_paper\code\iem-code\datasets_jpg\jpg"  # 替换为您实际的文件夹路径

# 获取所有jpg文件的名称
image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

# 计算训练集和测试集的大小
total_images = len(image_files)
train_size = int(0.7 * total_images)
test_size = total_images - train_size

# 随机分配图像到训练集和测试集
np.random.shuffle(image_files)
train_files = image_files[:train_size]
test_files = image_files[train_size:]

# 将文件名转换为索引（假设文件名为图像的唯一标识符）
train_indices = [int(f.split('.')[0].split('_')[-1]) for f in train_files]
test_indices = [int(f.split('.')[0].split('_')[-1]) for f in test_files]

# 将索引保存到MAT文件
setid = {'trnid': np.array(train_indices), 'tstid': np.array(test_indices)}
scipy.io.savemat('setid.mat', setid)

print("Total images:", total_images)
print("Training images:", len(train_files))
print("Testing images:", len(test_files))

print("Training indices:", train_indices)
print("Testing indices:", test_indices)

