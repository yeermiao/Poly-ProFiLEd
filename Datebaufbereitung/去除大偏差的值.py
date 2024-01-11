import os
import pandas as pd

root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'

def compare_and_delete(folder_path):
    # 获取文件夹中所有以 "Eigenspannungen.xlsx" 结尾的文件
    files = [f for f in os.listdir(folder_path) if f.endswith("Eigenspannungen.xlsx")]

    if len(files) != 3:
        # 如果文件数量不为 3，不进行对比操作
        return

    # 读取三个文件的第二行数据
    data_a = pd.read_excel(os.path.join(folder_path, files[0]), header=None, nrows=2).iloc[1, :]
    data_b = pd.read_excel(os.path.join(folder_path, files[1]), header=None, nrows=2).iloc[1, :]
    data_c = pd.read_excel(os.path.join(folder_path, files[2]), header=None, nrows=2).iloc[1, :]

    # 计算均值和标准差
    mean_a, std_a = data_a.mean(), data_a.std()
    mean_b, std_b = data_b.mean(), data_b.std()
    mean_c, std_c = data_c.mean(), data_c.std()

    # 均值加减2倍标准差的判断标准
    threshold = 2  # 根据实际情况调整阈值

    # 判断异常值
    outliers_a = (data_a > (mean_a + threshold * std_a)) | (data_a < (mean_a - threshold * std_a))
    outliers_b = (data_b > (mean_b + threshold * std_b)) | (data_b < (mean_b - threshold * std_b))
    outliers_c = (data_c > (mean_c + threshold * std_c)) | (data_c < (mean_c - threshold * std_c))

    if any(outliers_a) or any(outliers_b) or any(outliers_c):
        # 如果存在异常值，则删除该文件夹下的所有 xlsx 文件
        for file in files:
            os.remove(os.path.join(folder_path, file))
        print(f"Deleted files in {folder_path}")

# 遍历根目录下的所有参数组合文件夹
for subdir in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, subdir)
    if os.path.isdir(folder_path):
        compare_and_delete(folder_path)


