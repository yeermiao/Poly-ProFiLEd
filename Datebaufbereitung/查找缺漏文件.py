import os
import pandas as pd

def process_folders(root_path):
    results = []

    for param_folder in os.listdir(root_path):
        param_folder_path = os.path.join(root_path, param_folder)

        if os.path.isdir(param_folder_path):
            subfolders = ['a', 'b', 'c']
            counter = 0
            subfolder_numbers = {'a': 'fehlt', 'b': 'fehlt', 'c': 'fehlt'}

            for subfolder in subfolders:
                subfolder_path = os.path.join(param_folder_path, subfolder)
                asc_files = [f for f in os.listdir(subfolder_path) if f.endswith('.asc')]

                if asc_files:
                    subfolder_numbers[subfolder] = asc_files[0]
                    counter += 1

            result_row = [param_folder] + [subfolder_numbers[subfolder] for subfolder in subfolders] + [counter]
            results.append(result_row)

    return results

def save_to_excel(data, output_path):
    columns = ['Bearbeitungsparameter', 'a', 'b', 'c', 'Zähler']
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(output_path, index=False)

if __name__ == "__main__":
    root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'
    output_excel_path = r'F:\Users\yeerm\Desktop\result.xlsx'

    results_data = process_folders(root_directory)
    save_to_excel(results_data, output_excel_path)

    print("处理完成，结果保存在", output_excel_path)
