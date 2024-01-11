import os
import numpy as np
import pandas as pd
import ICP
import Feature_Extrahieren as FE
import Feature_Berechnen as FB
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.pyplot as plt



true_deformation_file = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\KSS_1_2670_40_016\a\KSS_1_2670_40_016_a.asc'
simulated_deformation_file = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\KSS_1_2670_40_016\a\Verformung.txt'





# region split_and_visualize
def split_and_visualize_asc_data_3d(file_path):
    # Daten aus der ASC-Datei laden
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    data = np.loadtxt(file_path, skiprows=1)  # 假设ASC文件使用空格分隔数据
    filtered_data = None
    if file_extension == ".asc":
        data = np.delete(data, 2, axis=1)
        original_filter = (data[:, 0] < 50) | (data[:, 0] > 100) | (data[:, 1] < 60) | (data[:, 1] > 85)
        additional_filter1 = (data[:, 0] >= 5) & (data[:, 0] <= 145) & (data[:, 1] >= 5) & (data[:, 1] <= 145)
        additional_filter2 = (data[:, 0] <= 75) | (data[:, 0] >= 80)
        combined_filter = original_filter & additional_filter2 & additional_filter1
        filtered_data = data[combined_filter]
    # Die Dateierweiterung '.txt' zeigt an, dass es sich um Daten handelt,
    # die aus ANSYS exportiert wurden.
    # Da die Koordinatensysteme von ANSYS und den CMM-Messdaten unterschiedlich sind,
    # ist eine Rotation und Translation erforderlich
    elif file_extension == ".txt":
        data = np.delete(data, 2, axis=1)
        theta = np.deg2rad(90)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        data = np.dot(data, rotation_matrix.T)
        data[:, 0] += 150
        filtered_data = data

    x_coordinates = filtered_data[:, 0]

    idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, z_value in filtered_data])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(idata)

    # 创建平面点云
    center = point_cloud.get_center()
    half_width = 75  # Hälfte der Breite in X-Richtung
    half_length = 75  # Hälfte der Länge in Y-Richtung
    step_x = 0.5  # Schrittweite in X-Richtung
    step_y = 0.5  # Schrittweite in Y-Richtung

    points = []
    for x in np.arange(center[0] - half_width + 0.5, center[0] + half_width - 0.5, step_x):
        for y in np.arange(center[1] - half_length + 0.5, center[1] + half_length - 0.5, step_y):
            point = [x, y, 0]
            points.append(point)

    plane = o3d.geometry.PointCloud()
    vertices = np.asarray(points)
    plane.points = o3d.utility.Vector3dVector(vertices)
    plane.estimate_normals()

    # 进行点云配准
    threshold = 0.5  # Schwellenwert für die Registrierung
    trans_init = np.identity(4)  # Anfangstransformation als Einheitsmatrix

    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud, plane, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # Daten in zwei Teile aufteilen basierend auf der X-Achsen-Koordinate
    # Wählt Daten aus, bei denen die X-Koordinate kleiner als 75 ist, und speichert sie in data_below_75.
    Ap1 = filtered_data[x_coordinates < 75]
    # Wählt Daten aus, bei denen die X-Koordinate größer oder gleich 80 ist, und speichert sie in data_above_80.
    Ap2 = filtered_data[x_coordinates >= 80]
    # Vereinheitlichung der X-Achse
    # Subtrahiert von allen Einträgen in der ersten Spalte (X-Koordinate) von data_above_80 den Wert 79.
    Ap2[:, 0] -= 73

    return Ap1, Ap2, file_name
# endregion


true_ap1, true_ap2, _ = split_and_visualize_asc_data_3d(true_deformation_file)
trans_true_ap1 = ICP.register_and_transform_point_cloud(true_ap1)
trans_true_ap2 = ICP.register_and_transform_point_cloud(true_ap2)
true_ap1_diagonal1, true_ap1_diagonal2 = FE.diagonale_feature_extrahieren(trans_true_ap1)
true_ap2_diagonal1, true_ap2_diagonal2 = FE.diagonale_feature_extrahieren(trans_true_ap2)

_, _, true_AP1_curve1_X = FB.fit_polynomial_curve(true_ap1_diagonal1, 3, 'X')
_, _, true_Ap1_curve2_X = FB.fit_polynomial_curve(true_ap1_diagonal2, 3, 'X')

_, _, true_AP1_curve1_Y = FB.fit_polynomial_curve(true_ap1_diagonal1, 3, 'Y')
_, _, true_Ap1_curve2_Y = FB.fit_polynomial_curve(true_ap1_diagonal2, 3, 'Y')


_, _, true_AP2_curve1_X = FB.fit_polynomial_curve(true_ap2_diagonal1, 3, 'X')
_, _, true_Ap2_curve2_X = FB.fit_polynomial_curve(true_ap2_diagonal2, 3, 'X')

_, _, true_AP2_curve1_Y = FB.fit_polynomial_curve(true_ap2_diagonal1, 3, 'Y')
_, _, true_Ap2_curve2_Y = FB.fit_polynomial_curve(true_ap2_diagonal2, 3, 'Y')


sim_ap1, sim_ap2, _ = split_and_visualize_asc_data_3d(simulated_deformation_file)
trans_sim_ap1 = ICP.register_and_transform_point_cloud(sim_ap1)
trans_sim_ap2 = ICP.register_and_transform_point_cloud(sim_ap2)
sim_ap1_diagonal1, sim_ap1_diagonal2 = FE.diagonale_feature_extrahieren(trans_sim_ap1)
sim_ap2_diagonal1, sim_ap2_diagonal2 = FE.diagonale_feature_extrahieren(trans_sim_ap2)

_, _, sim_AP1_curve1_X = FB.fit_polynomial_curve(sim_ap1_diagonal1, 3, 'X')
_, _, sim_Ap1_curve2_X = FB.fit_polynomial_curve(sim_ap1_diagonal2, 3, 'X')

_, _, sim_AP1_curve1_Y = FB.fit_polynomial_curve(sim_ap1_diagonal1, 3, 'Y')
_, _, sim_Ap1_curve2_Y = FB.fit_polynomial_curve(sim_ap1_diagonal2, 3, 'Y')


_, _, sim_AP2_curve1_X = FB.fit_polynomial_curve(sim_ap2_diagonal1, 3, 'X')
_, _, sim_Ap2_curve2_X = FB.fit_polynomial_curve(sim_ap2_diagonal2, 3, 'X')

_, _, sim_AP2_curve1_Y = FB.fit_polynomial_curve(sim_ap2_diagonal1, 3, 'Y')
_, _, sim_Ap2_curve2_Y = FB.fit_polynomial_curve(sim_ap2_diagonal2, 3, 'Y')


def plot_comparison_curves(true_curve1_X, true_curve2_X, sim_curve1_X, sim_curve2_X, filename):
    fig, ax = plt.subplots()
    ax.plot(true_curve1_X[:, 0], true_curve1_X[:, 1], label='True_Curve1_X', color='blue')
    ax.plot(true_curve2_X[:, 0], true_curve2_X[:, 1], label='True_Curve2_X', color='green')
    ax.plot(sim_curve1_X[:, 0], sim_curve1_X[:, 1], label='Sim_Curve1_X', color='red')
    ax.plot(sim_curve2_X[:, 0], sim_curve2_X[:, 1], label='Sim_Curve2_X', color='orange')

    # ax.set_xlabel('X-Achse')
    # ax.set_ylabel('Y-Achse')
    # ax.set_title('Vergleich der X-Kurven von True und Sim')
    # Hide axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ax.legend()

    # 设置外边框和网格
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.grid(True, color='black')  # 添加网格线

    fixed_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\\'
    full_path = os.path.join(fixed_path, filename)
    plt.savefig(full_path, format='eps', dpi=300, transparent=True)
    plt.show()


plot_comparison_curves(true_AP1_curve1_X, true_Ap1_curve2_X, sim_AP1_curve1_X, sim_Ap1_curve2_X, 'Ap1_X_Ansicht')
plot_comparison_curves(true_AP1_curve1_Y, true_Ap1_curve2_Y, sim_AP1_curve1_Y, sim_Ap1_curve2_Y, 'Ap1_Y_Ansicht')

plot_comparison_curves(true_AP2_curve1_X, true_Ap2_curve2_X, sim_AP2_curve1_X, sim_Ap2_curve2_X, 'Ap2_X_Ansicht')
plot_comparison_curves(true_AP2_curve1_Y, true_Ap2_curve2_Y, sim_AP2_curve1_Y, sim_Ap2_curve2_Y, 'Ap2_Y_Ansicht')