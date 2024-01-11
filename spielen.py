import joblib
import pandas as pd

# 指定.joblib文件的路径
model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_16_gegenlauf_0_σyy_mean_Random Forest_model.joblib'
# model_path = r'F:\Users\yeerm\Desktop\ML-Modelle\data_16_gegenlauf_0_σxx_mean_SVM_model.joblib'
input_columns = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']

# 加载模型
loaded_model = joblib.load(model_path)

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
input_data = {'Schnittgeschwindigkeit': 500, 'Eingriffsbreite': 8, 'Zahnvorschub': 0.1, 'Eingriffstiefe': 1.55}
new_input = pd.DataFrame([input_data], columns=input_columns)
prediction = loaded_model.predict(new_input)
print(prediction)