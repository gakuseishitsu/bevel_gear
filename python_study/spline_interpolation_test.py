import numpy as np
from scipy.interpolate import CubicSpline

# 5点の3次元空間の点
points = np.array([
    [-0.9, 0.81, -0.9],
    [-0.5, 0.25, -0.5],
    [-0.2, 0.04, -0.2],
    [0.1, 0.01, 0.1],
    [0.4, 0.16, 0.4],
    [0.95, 0.95*0.95, 0.95]
])

# 5点目までのu座標
u = np.array([-0.9, -0.5, -0.2, 0.1, 0.4, 0.95])

# それぞれの次元ごとにスプライン補間を計算
spline_x = CubicSpline(u, points[:,0], bc_type='natural')
spline_y = CubicSpline(u, points[:,1], bc_type='natural')
spline_z = CubicSpline(u, points[:,2], bc_type='natural')

# u=0とu=1での座標値を求める
u0_coords = np.array([spline_x(-1.0), spline_y(-1.0), spline_z(-1.0)])
u1_coords = np.array([spline_x(1.0), spline_y(1.0), spline_z(1.0)])

print("u=-1 での座標値:", u0_coords)
print("u=1 での座標値:", u1_coords)
