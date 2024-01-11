import os
import shutil
import re
from collections import deque

# Der Hauptzweck dieses Abschnitts des Codes besteht darin,
# basierend auf CMM-Messdaten, die Daten zu klassifizieren,
# insbesondere für jede Parameterkombination mit den drei Wiederholungen a, b und c.
# Zweitens, basierend auf den Ergebnissen der ersten Klassifizierung,
# werden die entsprechenden Spannungsüberwachungsdaten in die Ordner
# a, b und c der entsprechenden Parameterkombinationen klassifiziert.
# Pfad zum Messdatenordner
messdaten_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'
einzelne_res_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Einzelne_res_Dateien'

# Alle Unterordner im Messdatenordner durchsuchen
for root, dirs, files in os.walk(messdaten_directory):
    processing_done = False

    for dir_name in dirs:
        subdir_path = os.path.join(root, dir_name)
        asc_files = [file for file in os.listdir(subdir_path) if file.endswith('.asc')]

        # 创建a, b, c文件夹
        for letter in ['a', 'b', 'c']:
            letter_dir = os.path.join(subdir_path, letter)
            os.makedirs(letter_dir, exist_ok=True)

        # 将相应的.asc文件移到对应的文件夹
        for asc_file in asc_files:
            asc_number = asc_file[0]  # 获取编号
            letter = asc_file[-5]  # 获取a, b, c
            source_path = os.path.join(subdir_path, asc_file)
            destination_path = os.path.join(subdir_path, letter, asc_file)
            shutil.move(source_path, destination_path)

        processing_done = True  # 标记为处理完成

    if processing_done:
        break  # 如果已经处理完毕，终止遍历

print("文件已归类完成。")


for xlsx_file in os.listdir(einzelne_res_directory):
    if xlsx_file.endswith('.xlsx'):
        # 提取关键信息
        match = re.search(r'KSS_[01]_\d+_\d+_[,0-9]*_[abc]_ap[12]_res.xlsx', xlsx_file)

        if match:
            key_info = re.search(r'KSS_[01]_\d+_\d+_[,0-9]*', xlsx_file).group(0).replace(',', '')  # 提取关键信息，去除逗号
            abc_match = re.search(r'[abc]', xlsx_file).group(0)

            # 遍历Abweichungen_und_Eigenspannung下的文件夹
            folder_found = False  # 是否找到匹配的文件夹
            for root, dirs, files in os.walk(messdaten_directory):
                if folder_found:
                    break  # 如果已经找到匹配的文件夹，跳出遍历
                for dir_name in dirs:
                    if key_info in dir_name:
                        subdir_path = os.path.join(root, dir_name)
                        a_dir = os.path.join(subdir_path, 'a')
                        b_dir = os.path.join(subdir_path, 'b')
                        c_dir = os.path.join(subdir_path, 'c')
                        # 获取a, b, c编号
                        a_match = re.search(r'a', abc_match)
                        b_match = re.search(r'b', abc_match)
                        c_match = re.search(r'c', abc_match)

                        if a_match:
                            a_dest = os.path.join(a_dir, xlsx_file)
                            shutil.move(os.path.join(einzelne_res_directory, xlsx_file), a_dest)
                            folder_found = True  # 标记为已找到匹配的文件夹
                            break
                        elif b_match:
                            b_dest = os.path.join(b_dir, xlsx_file)
                            shutil.move(os.path.join(einzelne_res_directory, xlsx_file), b_dest)
                            folder_found = True  # 标记为已找到匹配的文件夹
                            break
                        elif c_match:
                            c_dest = os.path.join(c_dir, xlsx_file)
                            shutil.move(os.path.join(einzelne_res_directory, xlsx_file), c_dest)
                            folder_found = True  # 标记为已找到匹配的文件夹
                            break

                if not folder_found:
                    # 没找到匹配的文件夹，跳出外层循环
                    break

print("处理完成。")





