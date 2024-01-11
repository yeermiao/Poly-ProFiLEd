import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 指定.joblib文件的路径
model_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\best_models\D50_low_Vc_gleichlauf_1_Ap1_σyy_mean_Random Forest_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub']

# 加载模型
loaded_model = joblib.load(model_path)

# 固定值
zahnvorschub = 0.2

# 变化范围
schnittgeschwindigkeiten = np.arange(500, 901, 50)  # 从500到900，步长为50
eingriffsbreiten = np.arange(25, 41, 1.9)  # 从25到40，步长为1.5

# 参数组合
parameter_combinations = [(geschw, breite, zahnvorschub) for geschw in schnittgeschwindigkeiten for breite in eingriffsbreiten]

# 预测结果
predictions = []

for geschw, breite, zahn in parameter_combinations:
    input_data = {'Schnittgeschwindigkeit': geschw, 'Eingriffsbreite': breite, 'Zahnvorschub': zahn}
    new_input = pd.DataFrame([input_data], columns=input_columns)
    prediction = loaded_model.predict(new_input)
    predictions.append(prediction[0])

# 转换为 NumPy 数组
geschws, breiten = np.meshgrid(schnittgeschwindigkeiten, eingriffsbreiten)  # 修改这里的变量名
predictions = np.array(predictions).reshape(geschws.shape)

# 绘制二维图像
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(geschws, breiten, predictions, cmap='viridis')
# 添加颜色条
# fig.colorbar(surf, ax=ax, orientation='vertical', pad=0.2)

# 设置图表标题和轴标签
# ax.set_title('Predictions vs. Schnittgeschwindigkeit and Eingriffsbreite')
# ax.set_xlabel('Schnittgeschwindigkeit')
# ax.set_ylabel('Eingriffsbreite')
# ax.set_zlabel('Prognostiziertes σxx')
ax.view_init(azim=220, elev=30)

# 显示图像
plt.show()
