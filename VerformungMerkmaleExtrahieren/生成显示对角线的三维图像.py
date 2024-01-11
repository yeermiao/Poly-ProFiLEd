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

# region ICP_txt_version
def visualize_and_return_registered_point_cloud_txt(file_path):
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)

    # 随机抽样一半的数据点
    sampled_data = data[::6]

    idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, _, z_value in sampled_data])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(idata)

    center = point_cloud.get_center()
    half_width = 75
    half_length = 75
    step_x = 0.5
    step_y = 0.5

    points = []
    for x in np.arange(center[0] - half_width + 0.5, center[0] + half_width - 0.5, step_x):
        for y in np.arange(center[1] - half_length + 0.5, center[1] + half_length - 0.5, step_y):
            point = [x, y, 0]
            points.append(point)

    plane = o3d.geometry.PointCloud()
    vertices = np.asarray(points)
    plane.points = o3d.utility.Vector3dVector(vertices)
    plane.estimate_normals()

    threshold = 0.5
    trans_init = np.identity(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud, plane, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_title('')

    return points

# endregion

# region ICP
def visualize_and_return_registered_point_cloud(file_path):

    data = np.loadtxt(file_path, skiprows=1)
    # data = or_data[::2]

    original_filter = (data[:, 0] < 50) | (data[:, 0] > 100) | (data[:, 1] < 60) | (data[:, 1] > 85)
    additional_filter2 = (data[:, 0] <= 75) | (data[:, 0] >= 80)
    combined_filter = original_filter & additional_filter2
    filtered_data = data[combined_filter]

    idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, _, z_value in filtered_data])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(idata)

    center = point_cloud.get_center()
    half_width = 75
    half_length = 75
    step_x = 0.5
    step_y = 0.5

    points = []
    for x in np.arange(center[0] - half_width + 0.5, center[0] + half_width - 0.5, step_x):
        for y in np.arange(center[1] - half_length + 0.5, center[1] + half_length - 0.5, step_y):
            point = [x, y, 0]
            points.append(point)

    plane = o3d.geometry.PointCloud()
    vertices = np.asarray(points)
    plane.points = o3d.utility.Vector3dVector(vertices)
    plane.estimate_normals()

    threshold = 0.5
    trans_init = np.identity(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud, plane, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transformed_point_cloud = point_cloud.transform(reg_p2p.transformation)
    points = np.asarray(transformed_point_cloud.points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_title('')

    return points
# endregion

# region extract diagonals
def extract_diagonals(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    diagonal_point1 = [min_x, min_y]
    diagonal_point2 = [max_x, max_y]

    diagonal_point3 = [min_x, max_y]
    diagonal_point4 = [max_x, min_y]

    x_values = np.arange(min_x, max_x + 0.5, 1)
    diagonal_points1 = []
    diagonal_points2 = []

    for x_value in x_values:
        y_value_diagonal1 = (diagonal_point2[1] - diagonal_point1[1]) / (diagonal_point2[0] - diagonal_point1[0]) * (
                    x_value - diagonal_point1[0]) + diagonal_point1[1]
        y_value_diagonal2 = (diagonal_point4[1] - diagonal_point3[1]) / (diagonal_point4[0] - diagonal_point3[0]) * (
                    x_value - diagonal_point3[0]) + diagonal_point3[1]

        distances1 = np.sqrt((x - x_value) ** 2 + (y - y_value_diagonal1) ** 2)
        distances2 = np.sqrt((x - x_value) ** 2 + (y - y_value_diagonal2) ** 2)

        close_points1 = np.where(distances1 <= 1)
        if len(close_points1[0]) > 0:
            diagonal_points1.append([x[close_points1[0][0]], y[close_points1[0][0]], z[close_points1[0][0]]])

        close_points2 = np.where(distances2 <= 1)
        if len(close_points2[0]) > 0:
            diagonal_points2.append([x[close_points2[0][0]], y[close_points2[0][0]], z[close_points2[0][0]]])

    diagonal_points1 = np.array(diagonal_points1)
    diagonal_points2 = np.array(diagonal_points2)

    return diagonal_points1, diagonal_points2
# endregion

# region diagonale_feature
def diagonale_feature_extrahieren(data, filename):

    # Extrahieren der X-, Y- und Z-Koordinaten aus den Daten
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    Ap1 = data[x < 75]
    # Wählt Daten aus, bei denen die X-Koordinate größer oder gleich 80 ist, und speichert sie in data_above_80.
    Ap2 = data[x >= 80]

    diagonal_points1_ap1, diagonal_points2_ap1 = extract_diagonals(Ap1)
    diagonal_points1_ap2, diagonal_points2_ap2 = extract_diagonals(Ap2)

    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1, label='Point Cloud')
    ax.scatter(diagonal_points1_ap1[:, 0], diagonal_points1_ap1[:, 1], diagonal_points1_ap1[:, 2], c='red', s=10,
               label='Diagonal 1 Points (Ap1)')
    ax.scatter(diagonal_points2_ap1[:, 0], diagonal_points2_ap1[:, 1], diagonal_points2_ap1[:, 2], c='blue', s=10,
               label='Diagonal 2 Points (Ap1)')
    ax.scatter(diagonal_points1_ap2[:, 0], diagonal_points1_ap2[:, 1], diagonal_points1_ap2[:, 2], c='red', s=10,
               label='Diagonal 1 Points (Ap2)')
    ax.scatter(diagonal_points2_ap2[:, 0], diagonal_points2_ap2[:, 1], diagonal_points2_ap2[:, 2], c='blue', s=10,
               label='Diagonal 2 Points (Ap2)')
    # ax.set_xlabel('X Axis (mm)')
    # ax.set_ylabel('Y Axis (mm)')
    # ax.set_zlabel('Z Axis (mm)')
    # plt.title('Twisted Rectangle with Diagonals (3D)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(elev=60, azim=60)
    fixed_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Bilder\\'
    full_path = os.path.join(fixed_path, filename)
    plt.savefig(full_path, format='jpg', dpi=300, transparent=True)

    plt.show()


# endregion

trans_data = visualize_and_return_registered_point_cloud(true_deformation_file)
diagonale_feature_extrahieren(trans_data, 'true_deformation')


# trans_data = visualize_and_return_registered_point_cloud_txt(simulated_deformation_file)
# diagonale_feature_extrahieren(trans_data, 'sim_deformation')