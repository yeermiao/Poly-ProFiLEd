import os
import pandas as pd
import numpy as np

# 设置根目录路径
root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'

# 创建一个空的DataFrame来存储数据
result_df = pd.DataFrame(columns=['Parameterkombination', 'avg. △z Ap1_X', 'std. △z Ap1_X',
                                  'avg. △z Ap1_Y', 'std. △z Ap1_Y',
                                  'avg. △z Ap2_X', 'std. △z Ap2_X',
                                  'avg. △z Ap2_Y', 'std. △z Ap2_Y', 'Anzahl der Dateien'])

def process_parameter_combinations(root_directory):
    global result_df
    # Durchlaufe die Ordner für jede Parameterkombination im Stammverzeichnis
    for param_folder in os.listdir(root_directory):
        param_path = os.path.join(root_directory, param_folder)
        if os.path.isdir(param_path):
            avg_data = {'Parameterkombination': param_folder}
            data_sum = [0, 0, 0, 0]  # Initialisiere Summen für die vier Spalten
            data_sumsq = [0, 0, 0, 0]  # Initialisiere Summen der Quadrate für die Standardabweichung
            count = 0  # Initialisiere Zähler für die Anzahl der Dateien
            # Durchlaufe die a, b, c-Unterordner für jede Parameterkombination
            for sub_folder in os.listdir(param_path):
                sub_path = os.path.join(param_path, sub_folder)
                if os.path.isdir(sub_path):
                    for filename in os.listdir(sub_path):
                        if filename.endswith("Verformungsdaten.xlsx"):
                            file_path = os.path.join(sub_path, filename)
                            data_df = pd.read_excel(file_path, header=None, skiprows=5)
                            selected_columns = [1, 2, 3, 4]
                            data_sum = [data_sum[i] + data_df.iloc[0, selected_columns[i]] for i in range(4)]
                            data_sumsq = [data_sumsq[i] + data_df.iloc[0, selected_columns[i]]**2 for i in range(4)]
                            count += 1
            # Wenn Daten vorhanden sind, berechne den Durchschnitt und die Standardabweichung
            if count > 0:
                avg_data['avg. △z Ap1_X'] = data_sum[0] / count
                avg_data['std. △z Ap1_X'] = np.sqrt(data_sumsq[0] / count - (data_sum[0] / count)**2)
                avg_data['avg. △z Ap1_Y'] = data_sum[1] / count
                avg_data['std. △z Ap1_Y'] = np.sqrt(data_sumsq[1] / count - (data_sum[1] / count)**2)
                avg_data['avg. △z Ap2_X'] = data_sum[2] / count
                avg_data['std. △z Ap2_X'] = np.sqrt(data_sumsq[2] / count - (data_sum[2] / count)**2)
                avg_data['avg. △z Ap2_Y'] = data_sum[3] / count
                avg_data['std. △z Ap2_Y'] = np.sqrt(data_sumsq[3] / count - (data_sum[3] / count)**2)
                avg_data['Anzahl der Dateien'] = count
                result_df = pd.concat([result_df, pd.DataFrame(avg_data, index=[0])], ignore_index=True)

# Beginne mit der Verarbeitung des Stammverzeichnisses und seiner Unterordner
process_parameter_combinations(root_directory)

# Speichere die Ergebnisse in einer separaten xlsx-Datei
result_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Verformung_Ergebnisse.xlsx'
result_df.to_excel(result_path, index=False)

print("Datenverarbeitung abgeschlossen, Ergebnisse wurden in", result_path, "gespeichert")











