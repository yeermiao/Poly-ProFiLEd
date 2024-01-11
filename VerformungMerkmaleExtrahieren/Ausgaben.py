import os
import numpy as np
import pandas as pd
import ICP
import Feature_Extrahieren as FE
import Feature_Berechnen as FB
from tqdm import tqdm
import Eigenspannungen_Berechnung as EB
import matplotlib.pyplot as plt


# Der Hauptzweck dieses Abschnitts des Codes besteht darin, alle .asc- oder .txt-Dateien
# in dem angegebenen Verzeichnispfad zu durchsuchen.
# Für jede gefundene Datei wird die Funktion process_asc_file aufgerufen,
# um Merkmalsdaten zu extrahieren, diese zu verarbeiten,
# neue Merkmalsdaten zu generieren und die Merkmalsgrafiken im entsprechenden Verzeichnis zu speichern
# Funktion zur Verarbeitung einer einzelnen ASC-Datei und Rückgabe der Ausgabe
def process_asc_file(file_path, folder_path):
    # Schritt 1: Daten aus ASC-Datei laden und spalten
    ap1, ap2, file_name = ICP.split_and_visualize_asc_data_3d(file_path, folder_path)
    # Schritt 2: Punktewolken registrieren und transformieren
    trans_ap1 = ICP.register_and_transform_point_cloud(ap1)
    trans_ap2 = ICP.register_and_transform_point_cloud(ap2)
    # Schritt 3: Diagonale Merkmale extrahieren
    ap1_diagonal1, ap1_diagonal2 = FE.diagonale_feature_extrahieren(trans_ap1)
    ap2_diagonal1, ap2_diagonal2 = FE.diagonale_feature_extrahieren(trans_ap2)
    # Schritt 4: Kurvenvergleiche durchführen
    ap1_x_ansicht = FB.compare_curves(ap1_diagonal1, ap1_diagonal2, 3, 'X', 'AP1', file_name, folder_path)
    ap1_y_ansicht = FB.compare_curves(ap1_diagonal1, ap1_diagonal2, 3, 'Y', 'AP1', file_name, folder_path)
    ap2_x_ansicht = FB.compare_curves(ap2_diagonal1, ap2_diagonal2, 3, 'X', 'AP2', file_name, folder_path)
    ap2_y_ansicht = FB.compare_curves(ap2_diagonal1, ap2_diagonal2, 3, 'Y', 'AP2', file_name, folder_path)

    # plt.figure(1)
    # plt.scatter(Ap1_Diagonal1[:, 0], Ap1_Diagonal1[:, 2], c='red', s=10, label='Diagonal 1')
    # plt.scatter(Ap1_Diagonal2[:, 0], Ap1_Diagonal2[:, 2], c='blue', s=10, label='Diagonal 1')
    # plt.ylim([-0.05, 0.05])
    # plt.title('Ap1 Diagonale Merkmale Y-Ansicht')
    # plt.xlabel('X Achse (mm)')
    # plt.ylabel('Z Achse (mm)')
    #
    # plt.figure(2)
    # plt.scatter(Ap1_Diagonal1[:, 1], Ap1_Diagonal1[:, 2], c='red', s=10, label='Diagonal 1')
    # plt.scatter(Ap1_Diagonal2[:, 1], Ap1_Diagonal2[:, 2], c='blue', s=10, label='Diagonal 1')
    # plt.ylim([-0.05, 0.05])
    # plt.title('Ap1 Diagonale Merkmale X-Ansicht')
    # plt.xlabel('Y Achse (mm)')
    # plt.ylabel('Z Achse (mm)')
    #
    # plt.figure(3)
    # plt.scatter(Ap2_Diagonal1[:, 0], Ap2_Diagonal1[:, 2], c='red', s=10, label='Diagonal 2')
    # plt.scatter(Ap2_Diagonal2[:, 0], Ap2_Diagonal2[:, 2], c='blue', s=10, label='Diagonal 2')
    # plt.ylim([-0.05, 0.05])
    # plt.title('Ap2 Diagonale Merkmale Y-Ansicht')
    # plt.xlabel('X Achse (mm)')
    # plt.ylabel('Z Achse (mm)')
    #
    # plt.figure(4)
    # plt.scatter(Ap2_Diagonal1[:, 1], Ap2_Diagonal1[:, 2], c='red', s=10, label='Diagonal 2')
    # plt.scatter(Ap2_Diagonal2[:, 1], Ap2_Diagonal2[:, 2], c='blue', s=10, label='Diagonal 2')
    # plt.ylim([-0.05, 0.05])
    # plt.title('Ap2 Diagonale Merkmale X-Ansicht')
    # plt.xlabel('Y Achse (mm)')
    # plt.ylabel('Z Achse (mm)')
    #
    # plt.show()

    # Ausgabe-Pfad für die Excel-Datei erstellen
    output_file_path = os.path.splitext(file_path)[0] + '_Verformung.xlsx'

    ap1_x_ansicht = np.array(ap1_x_ansicht)
    ap1_y_ansicht = np.array(ap1_y_ansicht)
    ap2_x_ansicht = np.array(ap2_x_ansicht)
    ap2_y_ansicht = np.array(ap2_y_ansicht)

    # Schritt 5: Ergebnisse in Excel-Datei schreiben
    output_data = pd.DataFrame({
                              'Ap1_X_Ansicht': ap1_x_ansicht,
                              'Ap1_Y_Ansicht': ap1_y_ansicht,
                              'Ap2_X_Ansicht': ap2_x_ansicht,
                              'Ap2_Y_Ansicht': ap2_y_ansicht
                               })
    column_names = ['Maximalwert_der_Verformung', 'Minimalwert_der_Verformung', 'Gesamtverformung']
    output_df = pd.concat([pd.Series(column_names), output_data], axis=1)

    output_df.to_excel(output_file_path, index=False)

# Funktion zur Verarbeitung eines Verzeichnisses mit ASC-Dateien
def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith((".asc", ".txt")):
                file_path = os.path.join(root, file)
                folder_path = root
                process_asc_file(file_path, folder_path)


def main():
    input_directory = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung'
    # input_directory = r"F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung\KSS_1_2670_40_016\c"
    process_directory(input_directory)
    EB.berechnen(input_directory)


if __name__ == "__main__":
    main()