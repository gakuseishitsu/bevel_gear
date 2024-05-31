import numpy as np

# 3次元ベクトルを格納する空のnp.arrayを作成
vectors = np.empty((0, 3), dtype=float)

# 例として、5つの3次元ベクトルをforループで追加していく
for i in range(5):
    # 仮の座標値
    x = i
    y = i * 2
    z = i * 3
    
    # ベクトルを追加
    new_vector = np.array([[x, y, z]])
    vectors = np.append(vectors, new_vector, axis=0)

print(vectors)
