import pandas as pd

# 读取文件
# file_path = r'F:\Users\yeerm\Desktop\ansys\Surf_Nodes.txt'
file_path = r'F:\Users\yeerm\Desktop\ansys\小板子表面节点.txt'
df = pd.read_csv(file_path, sep='\t')

# 根据条件填充新列
df['Sigemaxx'] = 0.0
df['Sigemayy'] = 0.0
df['Txy'] = 0.0

# 根据不同的条件更新新列的值
# condition1 = (df['X Location (mm)'] >= 0) & (df['X Location (mm)'] <= 300) & ((df['Y Location (mm)'] >= 0) & (df['Y Location (mm)'] <= 150) | (df['Y Location (mm)'] >= 350) & (df['Y Location (mm)'] <= 500))
# condition2 = (df['Y Location (mm)'] > 150) & (df['Y Location (mm)'] < 350)

Ap1 = df['X Location (mm)'] < 75
Ap2 = df['X Location (mm)'] > 80

# D16
df.loc[Ap1, ['Sigemaxx', 'Sigemayy', 'Txy']] = [2.8, 4.9, -0.34]
df.loc[Ap2, ['Sigemaxx', 'Sigemayy', 'Txy']] = [1.74, 7.24, 0.07]

# D50
# df.loc[condition1, ['Sigemaxx', 'Sigemayy', 'Txy']] = [-0.25, -1.40, -0.39]
# df.loc[condition2, ['Sigemaxx', 'Sigemayy', 'Txy']] = [-2.31, 3.48, -0.22]


# 打印结果
output_path = r'F:\Users\yeerm\Desktop\ansys\Surf_Nodes_Eigenspannungen_klein.txt'
df.to_csv(output_path, sep='\t', index=False)  # 这里使用制表符作为分隔符，也可以选择其他分隔符
