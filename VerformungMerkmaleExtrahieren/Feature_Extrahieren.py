import numpy as np
import matplotlib.pyplot as plt

# Der Hauptzweck dieses Abschnitts des Codes besteht darin,
# Merkmale aus den 3D-Punktwolken-Daten zu extrahieren, die aus CMM-Messungen stammen.
# Die extrahierten Merkmale sind die beiden diagonalen Linien.
# Die zurückgegebenen Werte sind Kollektionen von Koordinatenpunkten, die sich auf den Diagonalen befinden


def diagonale_feature_extrahieren(data):

    # Extrahieren der X-, Y- und Z-Koordinaten aus den Daten
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Finden der minimalen und maximalen Koordinatenwerte der Punktewolke
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Berechnen der Koordinaten von zwei Punkten auf den Diagonalen
    diagonal_point1 = [min_x, min_y]
    diagonal_point2 = [max_x, max_y]

    diagonal_point3 = [min_x, max_y]
    diagonal_point4 = [max_x, min_y]

    # Generieren einer Reihe von X-Koordinaten mit einem Abstand von 1
    x_values = np.arange(min_x, max_x + 0.5, 1)
    diagonal_points1 = []
    diagonal_points2 = []

    for x_value in x_values:
        # Berechnen der Y-Koordinaten auf den beiden Diagonalen für die gegebene X-Koordinate
        y_value_diagonal1 = (diagonal_point2[1] - diagonal_point1[1]) / (diagonal_point2[0] - diagonal_point1[0]) * (
                    x_value - diagonal_point1[0]) + diagonal_point1[1]
        y_value_diagonal2 = (diagonal_point4[1] - diagonal_point3[1]) / (diagonal_point4[0] - diagonal_point3[0]) * (
                    x_value - diagonal_point3[0]) + diagonal_point3[1]

        # Berechnen der euklidischen Distanzen zu den Diagonalen
        distances1 = np.sqrt((x - x_value) ** 2 + (y - y_value_diagonal1) ** 2)
        distances2 = np.sqrt((x - x_value) ** 2 + (y - y_value_diagonal2) ** 2)

        # Finden von Punkten, die nahe genug an Diagonale 1 liegen
        close_points1 = np.where(distances1 <= 1)
        if len(close_points1[0]) > 0:
            diagonal_points1.append([x[close_points1[0][0]], y[close_points1[0][0]], z[close_points1[0][0]]])

        # Finden von Punkten, die nahe genug an Diagonale 2 liegen
        close_points2 = np.where(distances2 <= 1)
        if len(close_points2[0]) > 0:
            diagonal_points2.append([x[close_points2[0][0]], y[close_points2[0][0]], z[close_points2[0][0]]])

    # Markieren der Punkte auf den Diagonalen in Rot
    diagonal_points1 = np.array(diagonal_points1)
    diagonal_points2 = np.array(diagonal_points2)

    # plt.figure(2)
    # plt.scatter(x, y, c=z, cmap='viridis', s=1)  # 这里使用z值来设置点的颜色
    # plt.scatter(diagonal_points1[:, 0], diagonal_points1[:, 1], c='red', s=10, label='Diagonal 1')
    # plt.scatter(diagonal_points2[:, 0], diagonal_points2[:, 1], c='red', s=10, label='Diagonal 2')
    # plt.title('Twisted Rectangle with Diagonals')
    # plt.xlabel('X Axis (mm)')
    # plt.ylabel('Y Axis (mm)')
    #
    # plt.figure(3)
    # ax = plt.axes(projection='3d')
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=1, label='Point Cloud')
    # ax.scatter(diagonal_points1[:, 0], diagonal_points1[:, 1], diagonal_points1[:, 2], c='red', s=10,
    #            label='Diagonal 1 Points')
    # ax.scatter(diagonal_points2[:, 0], diagonal_points2[:, 1], diagonal_points2[:, 2], c='red', s=10,
    #            label='Diagonal 2 Points')
    # ax.set_xlabel('X Axis (mm)')  # 更改标签和单位
    # ax.set_ylabel('Y Axis (mm)')
    # ax.set_zlabel('Z Axis (mm)')
    # plt.title('Twisted Rectangle with Diagonals (3D)')
    #
    # plt.figure(4)
    # plt.scatter(diagonal_points1[:, 0], diagonal_points1[:, 2], c='red', s=10, label='Diagonal 1')
    # plt.scatter(diagonal_points2[:, 0], diagonal_points2[:, 2], c='red', s=10, label='Diagonal 2')
    # plt.title('Twisted Rectangle with Diagonals (X and Z)')

    # plt.figure(5)
    # plt.scatter(x, y, c=z, cmap='viridis', s=1)  # 这里使用z值来设置点的颜色
    # plt.title('Verformung von oben')
    # plt.xlabel('X Axis (mm)')
    # plt.ylabel('Y Axis (mm)')
    #
    # plt.xlabel('X Axis (mm)')
    # plt.ylabel('Z Axis (mm)')
    #
    # plt.show()

    # Rückgabe der beiden Listen von Punkten auf den Diagonalen
    return diagonal_points1, diagonal_points2