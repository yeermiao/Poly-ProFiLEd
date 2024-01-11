import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 设置根目录
root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'

# 初始化数据集
result_df = pd.DataFrame(columns=['KSS/noKSS', 'Werkzeug', 'Fraesrichtung', 'Schnittgeschwindigkeit',
                                  'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe',
                                  'σxx_mean', 'norm_σxx_SD', 'σyy_mean', 'norm_σyy_SD', 'Txy_mean', 'norm_Txy_SD',
                                  'Verzug (X)', 'Verzug_SD (X)', 'Verzug (Y)', 'Verzug_SD (Y)'])

# 遍历根目录下的参数组合文件夹
for folder_name in os.listdir(root_directory):
    param_path = os.path.join(root_directory, folder_name)
    if folder_name.startswith("KSS"):
        # 解析文件夹名字
        parameters = folder_name.split("_")
        KSS_noKSS = parameters[0]
        werkzeug = "D16" if parameters[3] == "8" else "D50"
        fraesrichtung = parameters[1]
        schnittgeschwindigkeit = parameters[2]
        eingriffsbreite = parameters[3]
        zahnvorschub = parameters[4]
        eingriffstiefe1 = 1
        eingriffstiefe2 = 2
        data_sum_1 = [0, 0, 0, 0, 0, 0]  # Initialisiere Summen für die vier Spalten
        data_sumsq_1 = [0, 0, 0, 0, 0, 0]  # Initialisiere Summen der Quadrate für die Standardabweichung

        norm_data_sum_1 = [0, 0, 0, 0, 0, 0]  # Initialisiere Summen für die vier Spalten
        norm_data_sumsq_1 = [0, 0, 0, 0, 0, 0]  # Initialisiere Summen der Quadrate für die Standardabweichung

        data_sum_2 = [0, 0, 0, 0]  # Initialisiere Summen für die vier Spalten
        data_sumsq_2 = [0, 0, 0, 0]  # Initialisiere Summen der Quadrate für die Standardabweichung
        count = 0  # Initialisiere Zähler für die Anzahl der Dateien

        result_entry1 = {
            'KSS/noKSS': KSS_noKSS,
            'Werkzeug': werkzeug,
            'Fraesrichtung': fraesrichtung,
            'Schnittgeschwindigkeit': schnittgeschwindigkeit,
            'Eingriffsbreite': eingriffsbreite,
            'Zahnvorschub': zahnvorschub,
            'Eingriffstiefe': eingriffstiefe1,
            'σxx_mean': np.nan,
            'norm_σxx_SD': np.nan,
            'σyy_mean': np.nan,
            'norm_σyy_SD': np.nan,
            'Txy_mean': np.nan,
            'norm_Txy_SD': np.nan,
            'Verzug (X)': np.nan,
            'Verzug_SD (X)': np.nan,
            'Verzug (Y)': np.nan,
            'Verzug_SD (Y)': np.nan
        }

        # 初始化第二个数据项
        result_entry2 = {
            'KSS/noKSS': KSS_noKSS,
            'Werkzeug': werkzeug,
            'Fraesrichtung': fraesrichtung,
            'Schnittgeschwindigkeit': schnittgeschwindigkeit,
            'Eingriffsbreite': eingriffsbreite,
            'Zahnvorschub': zahnvorschub,
            'Eingriffstiefe': eingriffstiefe2,
            'σxx_mean': np.nan,
            'norm_σxx_SD': np.nan,
            'σyy_mean': np.nan,
            'norm_σyy_SD': np.nan,
            'Txy_mean': np.nan,
            'norm_Txy_SD': np.nan,
            'Verzug (X)': np.nan,
            'Verzug_SD (X)': np.nan,
            'Verzug (Y)': np.nan,
            'Verzug_SD (Y)': np.nan
        }
        scaler = StandardScaler()
        for sub_folder in os.listdir(param_path):
            sub_path = os.path.join(param_path, sub_folder)
            if os.path.isdir(sub_path):
                for filename in os.listdir(sub_path):
                    if filename.endswith("Eigenspannungen.xlsx"):
                        file_path = os.path.join(sub_path, filename)
                        data_df = pd.read_excel(file_path, header=None, skiprows=1)
                        selected_columns1 = [0, 1, 2, 3, 4, 5]

                        data_to_normalize = data_df.iloc[0, selected_columns1].values.reshape(1, -1)
                        normalized_data = scaler.fit_transform(data_to_normalize)
                        norm_data_sum_1 = [data_sum_1[i] + normalized_data[0, i] for i in range(6)]
                        norm_data_sumsq_1 = [data_sumsq_1[i] + normalized_data[0, i] ** 2 for i in range(6)]

                        data_sum_1 = [data_sum_1[i] + data_df.iloc[0, selected_columns1[i]] for i in range(6)]
                        data_sumsq_1 = [data_sumsq_1[i] + data_df.iloc[0, selected_columns1[i]] ** 2 for i in range(6)]
                        count += 1
                    elif filename.endswith("Verformung.xlsx"):  # 新增条件
                        file_path = os.path.join(sub_path, filename)
                        data_df = pd.read_excel(file_path, header=None, skiprows=3)
                        selected_columns2 = [1, 2, 3, 4]
                        data_sum_2 = [data_sum_2[i] + data_df.iloc[0, selected_columns2[i]] for i in range(4)]
                        data_sumsq_2 = [data_sumsq_2[i] + data_df.iloc[0, selected_columns2[i]] ** 2 for i in range(4)]
        if count > 0:
            result_entry1['σxx_mean'] = data_sum_1[0] / count
            result_entry1['norm_σxx_SD'] = np.sqrt(norm_data_sumsq_1[0] / count - (norm_data_sum_1[0] / count) ** 2)
            result_entry1['σyy_mean'] = data_sum_1[1] / count
            result_entry1['norm_σyy_SD'] = np.sqrt(norm_data_sumsq_1[1] / count - (norm_data_sum_1[1] / count) ** 2)
            result_entry1['Txy_mean'] = data_sum_1[2] / count
            result_entry1['norm_Txy_SD'] = np.sqrt(norm_data_sumsq_1[2] / count - (norm_data_sum_1[2] / count) ** 2)
            result_entry1['Verzug (X)'] = data_sum_2[0] / count
            result_entry1['Verzug_SD (X)'] = np.sqrt(data_sumsq_2[0] / count - (data_sum_2[0] / count) ** 2)
            result_entry1['Verzug (Y)'] = data_sum_2[1] / count
            result_entry1['Verzug_SD (Y)'] = np.sqrt(data_sumsq_2[1] / count - (data_sum_2[1] / count) ** 2)

            result_entry2['σxx_mean'] = data_sum_1[3] / count
            result_entry2['norm_σxx_SD'] = np.sqrt(norm_data_sumsq_1[3] / count - (norm_data_sum_1[3] / count) ** 2)
            result_entry2['σyy_mean'] = data_sum_1[4] / count
            result_entry2['norm_σyy_SD'] = np.sqrt(norm_data_sumsq_1[4] / count - (norm_data_sum_1[4] / count) ** 2)
            result_entry2['Txy_mean'] = data_sum_1[5] / count
            result_entry2['norm_Txy_SD'] = np.sqrt(norm_data_sumsq_1[5] / count - (norm_data_sum_1[5] / count) ** 2)
            result_entry2['Verzug (X)'] = data_sum_2[2] / count
            result_entry2['Verzug_SD (X)'] = np.sqrt(data_sumsq_2[2] / count - (data_sum_2[2] / count) ** 2)
            result_entry2['Verzug (Y)'] = data_sum_2[3] / count
            result_entry2['Verzug_SD (Y)'] = np.sqrt(data_sumsq_2[3] / count - (data_sum_2[3] / count) ** 2)

            result_df = pd.concat([result_df, pd.DataFrame([result_entry1, result_entry2])], ignore_index=True)

        else:
            result_entry1['σxx_mean'] = 'NaN'
            result_entry1['norm_σxx_SD'] = 'NaN'
            result_entry1['σyy_mean'] = 'NaN'
            result_entry1['norm_σyy_SD'] = 'NaN'
            result_entry1['Txy_mean'] = 'NaN'
            result_entry1['norm_Txy_SD'] = 'NaN'
            result_entry1['Verzug (X)'] = 'NaN'
            result_entry1['Verzug_SD (X)'] = 'NaN'
            result_entry1['Verzug (Y)'] = 'NaN'
            result_entry1['Verzug_SD (Y)'] = 'NaN'

            result_entry2['σxx_mean'] = 'NaN'
            result_entry2['norm_σxx_SD'] = 'NaN'
            result_entry2['σyy_mean'] = 'NaN'
            result_entry2['norm_σyy_SD'] = 'NaN'
            result_entry2['Txy_mean'] = 'NaN'
            result_entry2['norm_Txy_SD'] = 'NaN'
            result_entry2['Verzug (X)'] = 'NaN'
            result_entry2['Verzug_SD (X)'] = 'NaN'
            result_entry2['Verzug (Y)'] = 'NaN'
            result_entry2['Verzug_SD (Y)'] = 'NaN'

            result_df = pd.concat([result_df, pd.DataFrame([result_entry1, result_entry2])], ignore_index=True)

output_file_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Zusammenfassung_der_berechneten_Restspannungen.xlsx'

# 保存结果到Excel文件
result_df.to_excel(output_file_path, index=False)

