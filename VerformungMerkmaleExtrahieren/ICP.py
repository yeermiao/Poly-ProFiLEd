import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

# Die Funktion dieses Abschnitts des Codes besteht darin,
# die ICP-Kalibrierung durchzuführen. Dabei handelt es sich um die Verarbeitung von Punktewolken,
# die durch CMM-Messungen generiert wurden.
# Aufgrund der unterschiedlichen Positionen und Orientierungen der realen Daten im dreidimensionalen Raum
# wird eine Anpassung der Lage durchgeführt,
# um die Effektivität des nachfolgenden Merkmalsextraktionsprozesses zu verbessern und Fehler zu minimieren


def split_and_visualize_asc_data_3d(file_path, folder_path):
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
    y_coordinates = filtered_data[:, 1]
    z_coordinates = filtered_data[:, 2]

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

    Plane_points = np.asarray(plane.points)

    # 进行点云配准
    threshold = 0.5  # Schwellenwert für die Registrierung
    trans_init = np.identity(4)  # Anfangstransformation als Einheitsmatrix

    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud, plane, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # Das hier gezeigte Bild zeigt die Originaldaten
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_coordinates, y_coordinates, z_coordinates, c='b', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Transformed 3D Data')
    # # plt.show()
    # plot_filename = f'{file_name}_Verformung.png'
    # plot_filepath = os.path.join(folder_path, plot_filename)
    # plt.savefig(plot_filepath)
    # plt.close()
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
    )
    plot_filename = f'{file_name}_Verformung.html'
    plot_filepath = os.path.join(folder_path, plot_filename)
    fig.write_html(plot_filepath)


    # Daten in zwei Teile aufteilen basierend auf der X-Achsen-Koordinate
    # Wählt Daten aus, bei denen die X-Koordinate kleiner als 75 ist, und speichert sie in data_below_75.
    Ap1 = filtered_data[x_coordinates < 75]
    # Wählt Daten aus, bei denen die X-Koordinate größer oder gleich 80 ist, und speichert sie in data_above_80.
    Ap2 = filtered_data[x_coordinates >= 80]
    # Vereinheitlichung der X-Achse
    # Subtrahiert von allen Einträgen in der ersten Spalte (X-Koordinate) von data_above_80 den Wert 79.
    Ap2[:, 0] -= 73

    return Ap1, Ap2, file_name


def register_and_transform_point_cloud(data):
    # Aus den ursprünglichen Daten mit den Abmessungen N*4 die benötigten X-, Y- und Z-Abweichungsspalten auswählen.
    idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, z_value in data])

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
    # print("Inlier RMSE:", reg_p2p.inlier_rmse)
    # print(reg_p2p.transformation)

    # Anwendung der Registrierungstransformation auf die Originalpunktwolke
    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # ax.scatter(x, y, z, label='Zielpunktewolke', c='r', marker='o', s=1)
    # x_plane = Plane_points[:, 0]
    # y_plane = Plane_points[:, 1]
    # z_plane = Plane_points[:, 2]
    # ax.scatter(x_plane, y_plane, z_plane, label='Kalibrierungspunktewolke', c='b', marker='^', s=1)
    # ax.legend()
    # ax.set_title('Punktwolkenkalibrierungsvisualisierung')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # plt.show()

    return points



