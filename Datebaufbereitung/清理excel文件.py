import os

root_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'

def delete_excel_files(directory):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            delete_excel_files(folder_path)  # 递归调用，遍历子目录
        else:
            if folder_name.endswith("Ergebnisse.xlsx"):
                file_path = os.path.join(directory, folder_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 调用函数来删除指定目录下的所有 Excel 文件
delete_excel_files(root_directory)
