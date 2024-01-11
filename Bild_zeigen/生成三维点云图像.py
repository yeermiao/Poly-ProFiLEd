import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 读取txt文件
file_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\KSS_1_800_8_016\a\Verformung.txt'
data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)

# 随机抽样一半的数据点
sampled_data = data[::5]

# 提取抽样后的坐标
x = sampled_data[:, 0]
y = sampled_data[:, 1]
z = sampled_data[:, 3]
# 创建三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c= z, cmap='viridis', s=1)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_xticklabels([])  # 隐藏X轴刻度数值
ax.set_yticklabels([])  # 隐藏Y轴刻度数值
ax.set_zticklabels([])
ax.set_title('')

# 显示图像
plt.show()
