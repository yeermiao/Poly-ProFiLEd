import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import open3d as o3d

def read_and_split_data(file_path):
    # 读取.asc文件
    data = np.loadtxt(file_path, skiprows=1)

    # 提取X坐标
    x = data[:, 0]

    # 根据X值分割数据
    ap1 = data[x <= 150]
    ap2 = data[(x > 150) & (x < 350)]
    ap1_star = data[x >= 350]

    return ap1, ap2, ap1_star


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

# 示例用法
file_path = r"F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Punkte_Bl1bis6\Deharde_3.asc"
ap1_data, ap2_data, ap1_star_data = read_and_split_data(file_path)
params1, _ = fit_quadratic_surface(ap1_star_data)
stresses1 = calculate_stresses(params1[3], params1[4], params1[5], 3)
print("\nAp1 star Stresses Calculated:")
print(stresses1)

params2, _ = fit_quadratic_surface(ap2_data)
stresses2 = calculate_stresses(params2[3], params2[4], params2[5], 2)
print("\nAp2 Stresses Calculated:")
print(stresses2)

params3, _ = fit_quadratic_surface(ap1_data)
stresses3 = calculate_stresses(params3[3], params3[4], params3[5], 4)
print("\nAp1 Stresses Calculated:")
print(stresses3)


