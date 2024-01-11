import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 指定.joblib文件的路径
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_50_low_Vc_gleichlauf_1_Txy_mean_Random Forest_model.joblib'
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_16_gegenlauf_0_σxx_mean_SVM_model.joblib'
model_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\best_models\D16_gegenlauf_0_Ap2_σxx_mean_Neural Network_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub']

# 加载模型
loaded_model = joblib.load(model_path)


# D16
schnittgeschwindigkeiten = np.arange(500, 1001, 50)
eingriffsbreite = 8  # 固定值
zahnvorschub = 0.2  # 固定值

# D50
# schnittgeschwindigkeiten = np.arange(500, 901, 50)
# eingriffsbreite = 25  # 固定值
# zahnvorschub = 0.2  # 固定值

# 预测结果
predictions = []

for schnittgeschwindigkeit in schnittgeschwindigkeiten:
    input_data = {'Schnittgeschwindigkeit': schnittgeschwindigkeit, 'Eingriffsbreite': eingriffsbreite, 'Zahnvorschub': zahnvorschub}
    new_input = pd.DataFrame([input_data], columns=input_columns)
    prediction = loaded_model.predict(new_input)
    predictions.append(prediction[0])

# 绘制二维图像

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(schnittgeschwindigkeiten, predictions, marker='o',linewidth=2.0)
# ax.set_title('Predictions vs. Zahnvorschub')
# ax.set_xlabel('Zahnvorschub')
# ax.set_ylabel('Prognostiziertes σxx')
ax.grid(True, linewidth=2.0)
ax.grid(color='black', linestyle='-', linewidth=2.0)
ax.spines['top'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
# ax.set_xticklabels([])
# ax.set_yticklabels([])

# output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\data_50_low_Vc_gleichlauf_1_Txy.eps'
# plt.savefig(output_path, format='eps', bbox_inches='tight')

# 显示图像
plt.show()
