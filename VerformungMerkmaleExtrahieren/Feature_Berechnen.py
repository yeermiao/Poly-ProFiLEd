import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial

# Der Hauptzweck dieses Abschnitts des Codes besteht darin,
# die Ergebnisse der Merkmalsextraktion zu verarbeiten.
# Konkret geht es um die Bearbeitung der beiden diagonalen Merkmale, die ausgewählt wurden.
# Dieser Prozess umfasst die Entfernung von Ausreißern und die Anpassung eines Polynoms dritten Grades,
# um die verschiedenen Merkmale zwischen den beiden diagonalen Linien zu generieren


def fit_polynomial_curve(data, degree, ansicht_type):
    # Hier wird je nach ansicht_type entweder die erste oder die zweite Spalte als X-Koordinaten
    # und die zweite Spalte als Y-Koordinaten aus den Daten extrahiert.
    # Also, um die beiden Diagonalen aus verschiedenen X- oder Y-Blickwinkeln zu vergleichen.
    if ansicht_type == 'Y':
        x = data[:, 0]  # Extrahiere die erste Spalte als X-Koordinaten
        y = data[:, 2]  # Extrahiere die zweite Spalte als Y-Koordinaten
        x_values = np.linspace(5, 75, 100)
        curve = data[:, [0, 2]]
    elif ansicht_type == 'X':
        x = data[:, 1]  # Extrahiere die zweite Spalte als X-Koordinaten
        y = data[:, 2]  # Extrahiere die zweite Spalte als Y-Koordinaten
        x_values = np.linspace(5, 145, 100)
        curve = data[:, [1, 2]]
    else:
        raise ValueError("Ungültiger Koordinatentyp. Verwenden Sie 'X' oder 'Y'.")

    # Ausgleiche/Berechne die Koeffizienten des Polynoms
    coefficients = np.polyfit(x, y, degree)
    return coefficients, x_values, curve


def evaluate_polynomial(coefficients, x):
    # Bewertet das Polynom für die gegebenen X-Werte
    y = np.polyval(coefficients, x)
    return y


def compare_curves(input1, input2, degree, ansicht_type, AP, file_name, folder_path):
    # Kurve ausgleichen
    coff1, x_values, curve1 = fit_polynomial_curve(input1, degree, ansicht_type)
    coff2, x_values, curve2 = fit_polynomial_curve(input2, degree, ansicht_type)

    y_values1 = evaluate_polynomial(coff1, x_values)
    y_values2 = evaluate_polynomial(coff2, x_values)

    # Berechne die Summe array der beiden Kurven
    sum_curve = np.concatenate((y_values1, y_values2))

    # Finde den maximalen und minimalen Wert der Summenkurve
    max_value_sum_curve = np.max(sum_curve)
    min_value_sum_curve = np.min(sum_curve)
    distance = max_value_sum_curve - min_value_sum_curve


    # Hier sind die Bilder, die die extrahierten diagonalen
    # Merkmalspunkte und die angepassten Kurven dieser Punkte zeigen
    # Dieser plt-Code wird verwendet, um Feature-Grafiken zu erstellen
    # und in den entsprechenden Ordnern zu speichern, ohne sie anzuzeigen.
    title = f'{file_name}_{AP}_{ansicht_type}_Ansicht'
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values1, label='Kurve 1', color='red')
    ax.plot(x_values, y_values2, label='Kurve 2', color='blue')
    ax.scatter(curve1[:, 0], curve1[:, 1], label='Ursprüngliche Daten 1', color='red')
    ax.scatter(curve2[:, 0], curve2[:, 1], label='Ursprüngliche Daten 2', color='blue')
    ax.legend()
    ax.set_ylabel('Z (mm)')

    if ansicht_type == 'X':
        ax.set_xlabel('Y (mm)')
    elif ansicht_type == 'Y':
        ax.set_xlabel('X (mm)')

    ax.set_title(title)
    ax.set_ylim([min_value_sum_curve - 0.05, max_value_sum_curve + 0.01])

    # 移除坐标轴和标题
    # ax.set_xticklabels([])  # 隐藏X轴刻度数值
    # ax.set_yticklabels([])  # 隐藏Y轴刻度数值

    ax.set_title('')
    ax.grid(True)

    # plt.show()

    plot_filename = f'{file_name}_{AP}_{ansicht_type}_Ansicht.png'
    plot_filepath = os.path.join(folder_path, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    result_list = [max_value_sum_curve, min_value_sum_curve, distance]

    # Gebe die Liste der Ausgaben zurück
    return result_list