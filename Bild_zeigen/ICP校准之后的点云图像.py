import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# 读取ASC文件
file_path = r"F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Abweichungen_und_Eigenspannung\KSS_0_1000_8_01\c\KSS_0_1000_8_01_c.asc"
data = np.loadtxt(file_path, skiprows=1)  # 忽略第一行标题

original_filter = (data[:, 0] < 50) | (data[:, 0] > 100) | (data[:, 1] < 60) | (data[:, 1] > 85)
# additional_filter1 = (data[:, 0] >= 5) & (data[:, 0] <= 145) & (data[:, 1] >= 5) & (data[:, 1] <= 145)
additional_filter2 = (data[:, 0] <= 75) | (data[:, 0] >= 80)
combined_filter = original_filter & additional_filter2 #& additional_filter1
filtered_data = data[combined_filter]

# 将数据存储到Open3D点云对象中
idata = np.asarray([[x_value, y_value, z_value] for x_value, y_value, _, z_value in filtered_data])
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

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]
ax.scatter(x, y, z, c=z,cmap='viridis', s=1)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
# ax.set_xticklabels([])  # 隐藏X轴刻度数值
# ax.set_yticklabels([])  # 隐藏Y轴刻度数值
# ax.set_zticklabels([])
# ax.set_axis_off()
ax.set_title('')
# mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
#     f'X: {sel.target[0]:.2f}\nY: {sel.target[1]:.2f}\nZ: {sel.target[2]:.2f}'))
plt.show()
