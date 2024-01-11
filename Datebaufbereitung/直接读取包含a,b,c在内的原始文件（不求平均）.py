import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 设置根目录
root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'

# 初始化数据集
result_df = pd.DataFrame(columns=['KSS/noKSS', 'Werkzeug', 'Fraesrichtung', 'Schnittgeschwindigkeit',
                                  'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe', 'Versuch',
                                  'σxx', 'σyy', 'Txy', 'Verzug (X)', 'Verzug (Y)'])

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

        result_entry1 = {
            'KSS/noKSS': KSS_noKSS,
            'Werkzeug': werkzeug,
            'Fraesrichtung': fraesrichtung,
            'Schnittgeschwindigkeit': schnittgeschwindigkeit,
            'Eingriffsbreite': eingriffsbreite,
            'Zahnvorschub': zahnvorschub,
            'Eingriffstiefe': eingriffstiefe1,
            'Versuch': np.nan,
            'σxx': np.nan,
            'σyy': np.nan,
            'Txy': np.nan,
            'Verzug (X)': np.nan,
            'Verzug (Y)': np.nan,
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
            'Versuch': np.nan,
            'σxx': np.nan,
            'σyy': np.nan,
            'Txy': np.nan,
            'Verzug (X)': np.nan,
            'Verzug (Y)': np.nan,
        }
        scaler = StandardScaler()
        for sub_folder in os.listdir(param_path):
            sub_path = os.path.join(param_path, sub_folder)
            if os.path.isdir(sub_path):
                for filename in os.listdir(sub_path):
                    if filename.endswith("Eigenspannungen.xlsx"):
                        result_entry1['Versuch'] = sub_folder
                        result_entry2['Versuch'] = sub_folder
                        file_path = os.path.join(sub_path, filename)
                        data_df = pd.read_excel(file_path, header=None, skiprows=1)
                        selected_columns1 = [0, 1, 2, 3, 4, 5]

                        result_entry1['σxx'] = data_df.iloc[0, selected_columns1[0]]
                        result_entry1['σyy'] = data_df.iloc[0, selected_columns1[1]]
                        result_entry1['Txy'] = data_df.iloc[0, selected_columns1[2]]

                        result_entry2['σxx'] = data_df.iloc[0, selected_columns1[3]]
                        result_entry2['σyy'] = data_df.iloc[0, selected_columns1[4]]
                        result_entry2['Txy'] = data_df.iloc[0, selected_columns1[5]]


                    elif filename.endswith("Verformung.xlsx"):  # 新增条件
                        file_path = os.path.join(sub_path, filename)
                        data_df = pd.read_excel(file_path, header=None, skiprows=3)
                        selected_columns2 = [1, 2, 3, 4]

                        result_entry1['Verzug (X)'] = data_df.iloc[0, selected_columns2[0]]
                        result_entry1['Verzug (Y)'] = data_df.iloc[0, selected_columns2[1]]

                        result_entry2['Verzug (X)'] = data_df.iloc[0, selected_columns2[2]]
                        result_entry2['Verzug (Y)'] = data_df.iloc[0, selected_columns2[3]]

                result_df = pd.concat([result_df, pd.DataFrame([result_entry1, result_entry2])], ignore_index=True)





output_file_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Zusammenfassung_σ_Verzug.xlsx'

# 保存结果到Excel文件
result_df.to_excel(output_file_path, index=False)