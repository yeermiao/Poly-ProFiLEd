import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import open3d as o3d


# Der Hauptzweck dieses Abschnitts des Codes besteht darin,
# basierend auf der Kirchhoff-Plattenbalkentheorie und 3D-Punktwolken-Daten,
# die mit einem CMM-Messsystem erfasst wurden, die Krümmung der Biegefläche zu berechnen.
# Anschließend werden die berechneten Biegekrümmungswerte verwendet, um die Restspannungen zu berechnen
# Ersetzen den Pfad zu den ASC-Dateien



def split_and_visualize_asc_data_3d(asc_file_path):
    # Daten aus der ASC-Datei laden
    # Da ein Teil der Oberfläche Dehnungsmessstreifen aufweist
    # müssen die Messdaten diesen Bereich umgehen, da er ungenau ist
    data = np.loadtxt(asc_file_path, skiprows=1)
    original_filter = (data[:, 0] < 50) | (data[:, 0] > 100) | (data[:, 1] < 60) | (data[:, 1] > 85)
    additional_filter1 = (data[:, 0] >= 5) & (data[:, 0] <= 145) & (data[:, 1] >= 5) & (data[:, 1] <= 145)
    additional_filter2 = (data[:, 0] <= 75) | (data[:, 0] >= 80)
    # 结合原始筛选条件和新筛选条件，使用逻辑与运算符 "&"
    combined_filter = original_filter & additional_filter1 & additional_filter2
    filtered_data = data[combined_filter]

    # Die ASC-Daten bestehen aus vier Spalten: X, Y, Z und Abweichung.
    # Hier wird die Koordinate X der ersten Spalte extrahiert.
    x_coordinates = filtered_data[:, 0]

    # Daten in zwei Teile aufteilen basierend auf der X-Achsen-Koordinate
    # Da die Einteilung der Werkstück-Schnitttiefe Ap anhand der X-Achsenkoordinaten erfolgt
    # und einen Bereich zwischen 75 und 80 umfasst, aufgrund des Fehlers im Schneidkantenradius.
    # Wählt Daten aus, bei denen die X-Koordinate kleiner als 75 ist, und speichert sie in data_below_75.
    data_below_75 = filtered_data[x_coordinates < 75]
    # Wählt Daten aus, bei denen die X-Koordinate größer oder gleich 80 ist, und speichert sie in data_above_80.
    data_above_80 = filtered_data[x_coordinates >= 80]
    # Vereinheitlichung der X-Achse
    # Subtrahiert von allen Einträgen in der ersten Spalte (X-Koordinate) von data_above_80 den Wert 79.
    # So können die X-Achsenkoordinaten beider Punktewolken jeweils von 0 beginnend neu geordnet werden
    data_above_80[:, 0] -= 73
    #
    # Extrahieren die Daten aus den ersten drei Spalten und zeigen das Punktwolkenbild
    # plt.figure(1)
    # x = filtered_data[:, 0]
    # y = filtered_data[:, 1]
    # z = filtered_data[:, 3]
    # ax = plt.axes(projection='3d')
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=1, label='Point Cloud')
    # ax.set_xlabel('X Axis (mm)')
    # ax.set_ylabel('Y Axis (mm)')
    # ax.set_zlabel('Z Axis (mm)')
    # plt.title('Twisted Rectangle with Diagonals (3D)')
    #
    #
    #
    # #Zeigen jeweils die Bilder der beiden separierten Punktewolken.
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(data_below_75[:, 0], data_below_75[:, 1], data_below_75[:, 2], c=data_below_75[:, 2], cmap='viridis',s=1)
    # ax1.set_title('Data Ap1')
    # ax1.set_xlabel('X-axis')
    # ax1.set_ylabel('Y-axis')
    # ax1.set_zlabel('Z-axis')
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(data_above_80[:, 0], data_above_80[:, 1], data_above_80[:, 2], c=data_above_80[:, 2], cmap='viridis',s=1)
    # ax2.set_title('Data Ap2')
    # ax2.set_xlabel('X-axis')
    # ax2.set_ylabel('Y-axis')
    # ax2.set_zlabel('Z-axis')
    #
    # plt.tight_layout()
    # plt.show()

    # Rückgabe der Punktewolken-Daten nach der Trennung für die beiden Schnitttiefen Ap1 und Ap2 im Array-Format
    return data_below_75, data_above_80, filtered_data


def calculate_stresses(A, B, C, thickness):
    # Die Berechnungskonstanten einstellen
    E = 70300
    v = 0.33
    h = thickness
    z0 = 1
    r = z0 * (h/2 - z0/2)

    # Hierbei ist zu beachten, dass die Berechnung der Biegemomentenrichtung
    # mit den Koordinaten des Punktwolken übereinstimmt, was bedeutet,
    # dass sie sich von der Richtung der Restspannungen unterscheidet,
    # die durch das Bohrlochverfahren ermittelt wurden.
    stresses = []
    K = (E * h**3)/(12 * (1 - v**2))
    Mx = K * (A + v * B)
    My = K * (B + v * A)
    Mxy = K * (1 - v) * C
    # Mx = -((E * h ** 3) / (12 * (1 - v ** 2))) * (A + v * B)
    # My = -((E * h ** 3) / (12 * (1 - v ** 2))) * (B + v * A)
    # Mxy = -((E * h ** 3) / (12 * (1 - v ** 2))) * (1 - v) * C

    # Hier werden die Koordinaten entsprechend umgewandelt,
    # um sie mit der Messrichtung abzustimmen.
    stress_x = My / r
    stress_y = Mx / r
    stress_txy = - Mxy / r

    stresses.append(stress_x)
    stresses.append(stress_y)
    stresses.append(stress_txy)

    return stresses


def quadratic_surface(data, a, b, c, A, B, C):
    x = data[0]
    y = data[1]
    return 0.5 * A * x**2 + 0.5 * B * y**2 + C * x * y + b * x + c * y + a


def fit_quadratic_surface(data):
    # 提取输入数据中的 X、Y、Z（或 Abweichung） 列
    x_data = data[:, 0]
    y_data = data[:, 1]
    z_data = data[:, 2]  # 假设 Z 列是要拟合的因变量
    xx = np.expand_dims(x_data, 0)
    yy = np.expand_dims(y_data, 0)
    ind_variables = np.append(xx, yy, axis=0)

    # 使用 curve_fit 进行拟合
    result = curve_fit(quadratic_surface, ind_variables, z_data, maxfev=10000)
    params = result[0]
    y_pre = quadratic_surface(ind_variables, *params)
    r2 = r2_score(z_data, y_pre)

    # 生成两张图像进行对比
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始数据散点图
    ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', s=1, label='Original Data')

    # 绘制拟合后的曲面
    x_mesh, y_mesh = np.meshgrid(np.linspace(x_data.min(), x_data.max(), 100),
                                 np.linspace(y_data.min(), y_data.max(), 100))
    xy_mesh = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    z_pred = quadratic_surface(xy_mesh, *params).reshape(x_mesh.shape)
    ax.plot_surface(x_mesh, y_mesh, z_pred, cmap='viridis', alpha=0.5, label='Fitted Surface')

    # 设置图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X-achse')
    ax.set_ylabel('Y-achse')
    ax.set_zlabel('Z-achse')

    # 显示图形
    plt.show()

    return params, r2


def register_and_transform_point_cloud(data):
    # Aus den ursprünglichen Daten mit den Abmessungen N*4 die benötigten X-, Y- und Z-Abweichungsspalten auswählen.
    idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, _, z_value in data])

    # Erstellen eines 3D-Punktwolkenobjekts "point_cloud" und Zuweisen der Koordinaten aus "idata" dazu.
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(idata)
    # Ermitteln des Zentrums der Punktwolke "point_cloud".
    center = point_cloud.get_center()

    half_width = 37.5  # Hälfte der Breite in X-Richtung
    half_length = 75  # Hälfte der Länge in Y-Richtung
    step_x = 0.5  # Schrittweite in X-Richtung
    step_y = 0.5  # Schrittweite in Y-Richtung

    # Generiere eine 3D-Punktwolke mit den Abmessungen 75*180 um das “center” herum.
    points = []
    for x in np.arange(center[0] - half_width + 0.5, center[0] + half_width - 0.5, step_x):
        for y in np.arange(center[1] - half_length + 0.5, center[1] + half_length - 0.5, step_y):
            point = [x, y, 0]
            points.append(point)

    #  Erstellen eines leeren Punktwolkenobjekts "plane".
    plane = o3d.geometry.PointCloud()

    # Konvertieren der Punktkoordinaten in ein NumPy-Array und Zuweisen zum Punktwolkenobjekt "plane".
    vertices = np.asarray(points)
    plane.points = o3d.utility.Vector3dVector(vertices)
    plane.estimate_normals()

    Plane_points = np.asarray(plane.points)

    # Visualization “Plane”
    # o3d.visualization.draw_geometries([plane])

    threshold = 0.5 # Schwellenwert für die Registrierung
    trans_init = np.identity(4) # Anfangstransformation als Einheitsmatrix

    # Führe die ICP-Registrierung zwischen der Punktwolke "point_cloud" und der Ebene "plane" durch.
    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud, plane, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    # Ausgabe des RMSE-Werts für die Inlier (übereinstimmenden Punkte)
    print("Inlier RMSE:", reg_p2p.inlier_rmse)
    # print(reg_p2p.transformation)

    # Anwendung der Registrierungstransformation auf die Originalpunktwolke
    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制 transformed_below_75
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # ax.scatter(x, y, z, label='Below 75', c='r', marker='o', s=1)
    #
    # x_above_80 = Plane_points[:, 0]
    # y_above_80 = Plane_points[:, 1]
    # z_above_80 = Plane_points[:, 2]
    # ax.scatter(x_above_80, y_above_80, z_above_80, label='Above 80', c='b', marker='^', s=1)
    #
    # ax.legend()
    #
    # # 添加标题和标签
    # ax.set_title('3D Point Cloud Visualization')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 显示图形
    # plt.show()

    return points


def face_fitting(filepath):
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".asc"):
                asc_file_path = os.path.join(root, file)
                data_below_75, data_above_80, filtered_data = split_and_visualize_asc_data_3d(asc_file_path)
                registered_data_below_75 = register_and_transform_point_cloud(data_below_75)
                registered_data_above_80 = register_and_transform_point_cloud(data_above_80)

                params, r2 = fit_quadratic_surface(registered_data_below_75)
                stresses_below_75 = calculate_stresses(params[3], params[4], params[5], 3)

                params, r2 = fit_quadratic_surface(registered_data_above_80)
                stresses_above_80 = calculate_stresses(params[3], params[4], params[5], 2)

                # 创建一个DataFrame来存储结果数据
                result_entry = {
                    "Ap1_x (Mpa)": stresses_below_75[0],
                    "Ap1_y (Mpa)": stresses_below_75[1],
                    "Ap1_Txy (Mpa)": stresses_below_75[2],
                    "Ap2_x (Mpa)": stresses_above_80[0],
                    "Ap2_y (Mpa)": stresses_above_80[1],
                    "Ap2_Txy (Mpa)": stresses_above_80[2]
                }
                print(result_entry)



if __name__ == "__main__":
    input_directory = r"F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung\KSS_1_1000_8_02\c"
    face_fitting(input_directory)