import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 指定.joblib文件的路径
model_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\best_models\D50_low_Vc_gleichlauf_1_Ap2_σxx_mean_Random Forest_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub']

# 加载模型
loaded_model = joblib.load(model_path)

# 固定值
schnittgeschwindigkeit = 900

# 变化范围
zahnvorschuebe = np.arange(0.1, 0.21, 0.01)
eingriffsbreiten = np.arange(25, 41, 1.5)  # 从25到40，步长为1

# 参数组合
parameter_combinations = [(zahn, breite) for zahn in zahnvorschuebe for breite in eingriffsbreiten]

# 预测结果
predictions = []

for zahn, breite in parameter_combinations:
    input_data = {'Schnittgeschwindigkeit': schnittgeschwindigkeit, 'Eingriffsbreite': breite, 'Zahnvorschub': zahn}
    new_input = pd.DataFrame([input_data], columns=input_columns)
    prediction = loaded_model.predict(new_input)
    predictions.append(prediction[0])

# 转换为 NumPy 数组
zahns, breiten = np.meshgrid(zahnvorschuebe, eingriffsbreiten)  # 修改这里的变量名
predictions = np.array(predictions).reshape(zahns.shape)

# 绘制二维图像
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(zahns, breiten, predictions, cmap='viridis')
# 添加颜色条
fig.colorbar(surf, ax=ax, orientation='vertical', pad=0.2)

# 设置图表标题和轴标签
# ax.set_title('Predictions vs. Zahnvorschub and Eingriffsbreite')
# ax.set_xlabel('Zahnvorschub')
# ax.set_ylabel('Eingriffsbreite')
# ax.set_zlabel('Prognostiziertes σxx')

# 隐藏坐标轴上的标签
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
ax.view_init(azim=120, elev=30)

# 显示图像
plt.show()
