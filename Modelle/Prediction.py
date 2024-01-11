import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# 指定.joblib文件的路径
model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_50_low_Vc_gleichlauf_1_Txy_mean_Random Forest_model.joblib'
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_16_gegenlauf_0_σxx_mean_SVM_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']

# 加载模型
loaded_model = joblib.load(model_path)
# region 双交叉因素
zahns = np.arange(0.1, 0.21, 0.01)
depths = np.arange(1, 2.1, 0.1)
parameter_combinations = [(zahn, depth) for zahn in zahns for depth in depths]

# 预测结果
predictions = []

for zahn, depth in parameter_combinations:
    input_data = {'Schnittgeschwindigkeit': 500, 'Eingriffsbreite': 25, 'Zahnvorschub': zahn, 'Eingriffstiefe': depth}
    new_input = pd.DataFrame([input_data], columns=input_columns)
    prediction = loaded_model.predict(new_input)
    predictions.append(prediction[0])

# 转换为 NumPy 数组
zahns, depths = np.meshgrid(zahns, depths)
predictions = np.array(predictions).reshape(zahns.shape)


# zahns = np.arange(0.1, 0.21, 0.05)
# widths = np.arange(25, 41, 5)  # 修改这里的范围为25到40
# # widths = np.array([25, 40])
# parameter_combinations = [(zahn, width) for zahn in zahns for width in widths]
#
# # 预测结果
# predictions = []
#
# for zahn, width in parameter_combinations:
#     input_data = {'Schnittgeschwindigkeit': 500, 'Eingriffsbreite': width, 'Zahnvorschub': zahn, 'Eingriffstiefe': 1}
#     new_input = pd.DataFrame([input_data], columns=input_columns)
#     prediction = loaded_model.predict(new_input)
#     predictions.append(prediction[0])
#
# # 转换为 NumPy 数组
# zahns, widths = np.meshgrid(zahns, widths)  # 修改这里的变量名
# predictions = np.array(predictions).reshape(zahns.shape)



# speeds = np.arange(500, 901, 50)
# depths = np.arange(1, 2.1, 0.1)
#
# # 预测结果
# predictions = []
#
# for speed in speeds:
#     for depth in depths:
#         input_data = {'Schnittgeschwindigkeit': speed, 'Eingriffsbreite': 8, 'Zahnvorschub': 0.1, 'Eingriffstiefe': depth}
#         new_input = pd.DataFrame([input_data], columns=input_columns)
#         prediction = loaded_model.predict(new_input)
#         predictions.append(prediction[0])
#
# # 转换为 NumPy 数组
# speeds, depths = np.meshgrid(speeds, depths)
# predictions = np.array(predictions).reshape(speeds.shape)


# 绘制三维图像
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(azim=30, elev=20)
surf = ax.plot_surface(zahns, depths, predictions, cmap='viridis')
# 添加颜色条
fig.colorbar(surf, ax=ax, orientation='vertical')
# 设置图表标题和轴标签
# ax.set_title('Predictions vs. Schnittgeschwindigkeit and Eingriffstiefe')
# ax.set_xlabel('Schnittgeschwindigkeit')
# ax.set_ylabel('Eingriffstiefe')
# ax.set_zlabel('Prognostiziertes σxx')
# ax.view_init(azim=30, elev=20)
output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\data_50_low_Vc_gleichlauf_1_Txy.eps'
plt.savefig(output_path, format='eps', bbox_inches='tight')

# 显示图像
plt.show()
# endregion

# region 单速度因素
# 准备新的输入数据（这里修改第一个参数的值）
# input_data = {'Schnittgeschwindigkeit': [], 'Eingriffsbreite': 8, 'Zahnvorschub': 0.1, 'Eingriffstiefe': 1}
# predictions = []
#
# # 逐步修改第一个参数的值，并记录预测结果
# for speed in range(500, 1001, 50):
#     input_data['Schnittgeschwindigkeit'] = speed
#     new_input = pd.DataFrame([input_data], columns=input_columns)
#     prediction = loaded_model.predict(new_input)
#     predictions.append(prediction[0])
#  endregion

# region 单切削深度因素
# input_data = {'Schnittgeschwindigkeit': 500, 'Eingriffsbreite': 8, 'Zahnvorschub': 0.1, 'Eingriffstiefe': []}
# predictions = []
#
# # 逐步修改第四个参数的值，并记录预测结果
# depths = np.arange(1, 2.1, 0.1)
# for depth in depths:  # 从1到2，包括2
#     input_data['Eingriffstiefe'] = depth
#     new_input = pd.DataFrame([input_data], columns=input_columns)
#     prediction = loaded_model.predict(new_input)
#     predictions.append(prediction[0])
#
#
# plt.figure(figsize=(8, 6))
# # 设置图形的全局属性
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 20
# plt.rcParams['axes.linewidth'] = 1.5
# plt.rcParams['grid.linewidth'] = 1
#
# # 将结果绘制成图表，并添加网格线
# # plt.plot(range(500, 1001, 50), predictions, marker='o')
# plt.plot(depths, predictions, marker='o')
# plt.xlabel('Eingriffstiefe (mm)')
# plt.ylabel('Prognostiziertes Txy (Mpa)')
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # plt.title('Predictions vs. Schnittgeschwindigkeit')
# plt.grid(True, color='black')  # 添加网格线
# plt.tick_params(axis='both', which='major', labelsize=18)
#
# output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\data_16_gegenlauf_0_Txy.eps'
# plt.savefig(output_path, format='eps', bbox_inches='tight')
# plt.show()
# endregion



