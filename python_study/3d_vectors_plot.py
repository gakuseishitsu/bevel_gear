import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3次元ベクトルの配列
vectors = np.array([
    [-0.9, 0.81, -0.9],
    [-0.5, 0.25, -0.5],
    [-0.2, 0.04, -0.2],
    [0.1, 0.01, 0.1],
    [0.4, 0.16, 0.4],
    [0.95, 0.95*0.95, 0.95]
])

# プロット用のデータを準備
x_coords = vectors[:, 0]
y_coords = vectors[:, 1]
z_coords = vectors[:, 2]

# 3次元プロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# プロット
ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

# 軸ラベルの設定
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# グリッドの表示
ax.grid(True)

# グラフの表示
plt.show()