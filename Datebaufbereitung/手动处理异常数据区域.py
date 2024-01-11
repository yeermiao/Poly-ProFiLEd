import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 .asc 文件
def read_asc_file(file_path):
    return np.loadtxt(file_path, skiprows=1, dtype=str)

# 修改数据的函数（自定义这部分）
def modify_data(data):
    condition = ((data[:, 0].astype(float) > 43) & (data[:, 0].astype(float) < 110) &
                 (data[:, 1].astype(float) > 60) & (data[:, 1].astype(float) < 84))
    data = data[~condition, :]
    return data


# 将修改后的数据保存回 .asc 文件（保留原始文件的列名行）
def save_modified_data(file_path, modified_data, header):
    np.savetxt(file_path, modified_data, fmt='%s', delimiter='\t', header=header, comments='')

# 指定 .asc 文件路径
file_path = (r"F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung\KSS_1_2670_40_016\c\KSS_1_2670_40_016_c.asc")

# 读取 .asc 文件
data = read_asc_file(file_path)

# 获取列名
header = np.genfromtxt(file_path, max_rows=1, dtype=str)

# 修改数据（根据需要自定义）
modified_data = modify_data(data)

# 保存修改后的数据回 .asc 文件
save_modified_data(file_path, modified_data, '\t'.join(header))

# 将数据转换为浮点数
modified_data_float = modified_data.astype(float)

# 三维图形展示
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(modified_data_float[:, 0], modified_data_float[:, 1], modified_data_float[:, 3], c=modified_data_float[:, 3], cmap='viridis',s=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()





