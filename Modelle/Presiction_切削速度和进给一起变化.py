import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 指定.joblib文件的路径
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_50_low_Vc_gleichlauf_1_Txy_mean_Random Forest_model.joblib'
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_16_gegenlauf_0_σxx_mean_SVM_model.joblib'
model_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\best_models\D50_low_Vc_gleichlauf_1_Ap1_σxx_mean_Random Forest_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub']

# 加载模型
loaded_model = joblib.load(model_path)

eingriffsbreite = 8  # 固定值
zahnvorschuebe = np.arange(0.1, 0.21, 0.01)  # 从0.1到0.2，步长为0.01
schnittgeschwindigkeiten = np.arange(500, 1001, 50)  # 从500到1000，步长为50
parameter_combinations = [(zahn, geschw) for zahn in zahnvorschuebe for geschw in schnittgeschwindigkeiten]

# 预测结果
predictions = []

for zahn, geschw in parameter_combinations:
    input_data = {'Schnittgeschwindigkeit': geschw, 'Eingriffsbreite': eingriffsbreite, 'Zahnvorschub': zahn}
    new_input = pd.DataFrame([input_data], columns=input_columns)
    prediction = loaded_model.predict(new_input)
    predictions.append(prediction[0])

# 转换为 NumPy 数组
zahns, schwinds = np.meshgrid(zahnvorschuebe, schnittgeschwindigkeiten)  # 修改这里的变量名
predictions = np.array(predictions).reshape(zahns.shape)

# 绘制二维图像
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(azim=30, elev=20)
surf = ax.plot_surface(zahns, schwinds, predictions, cmap='viridis')
# 添加颜色条
fig.colorbar(surf, ax=ax, orientation='vertical', pad=0.2)
# 设置图表标题和轴标签
# ax.set_title('Predictions vs. Schnittgeschwindigkeit and Eingriffstiefe')
# ax.set_xlabel('Schnittgeschwindigkeit')
# ax.set_ylabel('Eingriffstiefe')
# ax.set_zlabel('Prognostiziertes σxx')
# ax.view_init(azim=30, elev=20)
# output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\data_50_low_Vc_gleichlauf_1_Txy.eps'
# plt.savefig(output_path, format='eps', bbox_inches='tight')
# ax.tick_params(axis='x', labelsize=18)  # x轴标签字体大小
# ax.tick_params(axis='y', labelsize=18)  # y轴标签字体大小
# ax.tick_params(axis='z', labelsize=18)  # z轴标签字体大小
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
ax.view_init(azim=120, elev=30)
# 显示图像
plt.show()
# 取消坐标轴上的标签

# output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\data_50_low_Vc_gleichlauf_1_Txy.eps'
# plt.savefig(output_path, format='eps', bbox_inches='tight')
