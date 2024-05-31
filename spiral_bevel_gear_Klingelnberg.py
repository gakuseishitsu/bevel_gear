import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline


'''
input valiables of bevel gear
'''
Zp = 16 # num of tooth pinion
Zg = 40 # num of tooth gear
Sigma = 90 # deg / shaft angle
PCDp = 540 # mm / Pitch circle diameter of pinion
PCDg = 1350 # mm / Pitch circle diameter of gear
B = 185 # mm /Face width
rc = 450 # mm / cutter radious
alpha = 20 # deg / pressure angle
beta = 32 # deg / spral angle
Z0 = 5 # num of thread of cutter
module = 33.75#24.799
#Exb = 4.5 # mm / radious difference
Exb = 33.75

'''
input valiables for computing
'''
vector_dimention = 3
resolution_theta = 30
resolution_neu = 30
resolution_psi = 30

'''
caliculated valiables of bevel gear
'''
i = Zg / Zp # Reduction ratio
delta_g0 = np.degrees(np.arctan2(np.sin(np.radians(Sigma)), 1/i + np.cos(np.radians(Sigma)))) # deg / angle of gear
delta_p0 = Sigma - delta_g0 # deg / angle of pinion
Rm = PCDg / (2 * np.sin(np.radians(delta_g0))) # mm / mean radious of Imaginary crown gear
Zc = Zg / np.sin(np.radians(delta_g0)) # num of tooth Imaginary crown gear
Md = np.sqrt(np.power(Rm, 2) + np.power(rc, 2) - 2*Rm*rc*np.cos(np.radians(90.0-beta))) # mm / machine distance
q = Md / (1 + Z0/Zc) # mm
r = Md - q # mm
neu_Rm = np.degrees(np.arccos((np.power(Md, 2) + np.power(Rm, 2) - np.power(rc, 2))/(2*Md*Rm))) # deg / initial angle of neu

'''
print('i: ', i) # 2.5
print('delta_g0: ', delta_g0) # 68.2 deg
print('delta_p0: ', delta_p0) # 21.8 deg
print('Rm: ', Rm) # 728.0 mm
print('Zc: ', Zc) # 43.1
print('Md: ', Md) # 619.9 mm
print('q: ', q) # 555.5 mm
print('r: ', r) # 64.5 mm
print('neu_Rm', neu_Rm) # 38.0 deg
'''

'''
decide a range of neu
'''
neu_range_pos = 0.0
while True:
    Xc_theta = np.array([0, rc, 0])
    phi_neu = Md / r * neu_range_pos + (90.0 -beta)
    C_phi = np.array([ # cordinate transfomation matrix: Z axis rotation
        [np.cos(np.radians(phi_neu)), -1.0 * np.sin(np.radians(phi_neu)), 0],
        [np.sin(np.radians(phi_neu)), np.cos(np.radians(phi_neu)), 0],
        [0, 0, 1.0]
    ])
    D_neu = np.array([ # position of the cutter center
        -1.0 * Md * np.sin(np.radians(neu_range_pos - neu_Rm)),
        Md * np.cos(np.radians(neu_range_pos - neu_Rm)),
        neu_range_pos * 0
    ])
    X = np.matmul(C_phi, Xc_theta) + D_neu
    Ri = np.sqrt(np.power(X[0], 2) + np.power(X[1], 2))
    if Ri <= Rm - 0.5 * B:
        break
    neu_range_pos += 0.001

neu_range_neg = 0.0
while True:
    Xc_theta = np.array([0, rc, 0])
    phi_neu = Md / r * neu_range_neg + (90.0 -beta)
    C_phi = np.array([ # cordinate transfomation matrix: Z axis rotation
        [np.cos(np.radians(phi_neu)), -1.0 * np.sin(np.radians(phi_neu)), 0],
        [np.sin(np.radians(phi_neu)), np.cos(np.radians(phi_neu)), 0],
        [0, 0, 1.0]
    ])
    D_neu = np.array([ # position of the cutter center
        -1.0 * Md * np.sin(np.radians(neu_range_neg - neu_Rm)),
        Md * np.cos(np.radians(neu_range_neg - neu_Rm)),
        neu_range_neg * 0
    ])
    X = np.matmul(C_phi, Xc_theta) + D_neu
    Re = np.sqrt(np.power(X[0], 2) + np.power(X[1], 2))
    if Re >= Rm + 0.5 * B:
        break
    neu_range_neg -= 0.001

#print('neu_range_pos: ', neu_range_pos, ', Ri: ', Ri, 'Rm-0.5B: ', Rm-0.5*B)
#print('neu_range_neg: ', neu_range_neg, ', Re: ', Re, 'Rm+0.5B: ', Rm+0.5*B)
neu = np.linspace(neu_range_neg-1, neu_range_pos+1, resolution_neu) # deg

'''
X_neu_theta: Coordinate values of the imaginary crown gear tooth surface point cloud
'''
theta = np.linspace(-1.0 * (1.25 * module)/(np.cos(np.radians(alpha))), (1.0 * module)/(np.cos(np.radians(alpha))), resolution_theta) # mm
psi = np.linspace(-15, 15, resolution_psi) # deg

X_neu_theta = np.zeros((resolution_neu, resolution_theta, vector_dimention)) # inner side of cutter
Xdash_neu_theta = np.zeros((resolution_neu, resolution_theta, vector_dimention)) # outer side of cutter

for index_theta, theta_n in enumerate(theta):
    for index_neu, neu_n in enumerate(neu):
        Xc_theta = np.array([
            theta_n * 0, 
            theta_n * np.sin(np.radians(alpha)) + rc, 
            theta_n * np.cos(np.radians(alpha))
        ])
        Xcdash_theta = np.array([
            theta_n * 0, 
            theta_n * -1 * np.sin(np.radians(alpha)) + rc + Exb, 
            theta_n * np.cos(np.radians(alpha))
        ])
        phi_neu = Md / r * neu_n + (90.0 - beta)
        C_phi = np.array([ # cordinate transfomation matrix: Z axis rotation
            [np.cos(np.radians(phi_neu)), -1.0 * np.sin(np.radians(phi_neu)), 0],
            [np.sin(np.radians(phi_neu)), np.cos(np.radians(phi_neu)), 0],
            [0, 0, 1.0]
        ])
        D_neu = np.array([ # position of the cutter center
            -1.0 * Md * np.sin(np.radians(neu_n - neu_Rm)),
            Md * np.cos(np.radians(neu_n - neu_Rm)),
            neu_n * 0
        ])
        X_neu_theta[index_neu][index_theta] = np.matmul(C_phi, Xc_theta) + D_neu
        Xdash_neu_theta[index_neu][index_theta] = np.matmul(C_phi, Xcdash_theta) + D_neu
        #print(X_neu_theta[index_neu][index_theta][0], ', ', X_neu_theta[index_neu][index_theta][1], ', ', X_neu_theta[index_neu][index_theta][2])

'''
N_neu_theta: Calculation of normal vectors of imaginary crown gear
'''
def compute_normals(X):
    normals = np.zeros((resolution_neu, resolution_theta, vector_dimention))
    for u in range(resolution_neu - 1):
        for v in range(resolution_theta -1):
            p0 = X[u][v]
            p1 = X[u + 1][v]
            p2 = X[u][v + 1]

            v1 = p1 - p0
            v2 = p2 - p0

            normal = np.cross(v1, v2) # normal vector direction is same as the cross product of v1,v2
            normal /= np.linalg.norm(normal)
            normals[u][v] = normal
    
    # interpolation for boudary
    for u in range(resolution_neu - 1):
        normals[u][-1] = normals[u][-2]

    for v in range(resolution_theta - 1):
        normals[-1][v] = normals[-2][v]

    normals[-1][-1] = normals[-2][-2]
    return normals

N_neu_theta = compute_normals(X_neu_theta)
Ndash_neu_theta = compute_normals(Xdash_neu_theta)

'''
X_neu_theta_psi: imaginary crown gear coordinates in absolute coordinate system
'''
X_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
N_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
W_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))

Xdash_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
Ndash_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
Wdash_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))

for index_theta, theta_n in enumerate(theta):
    for index_neu, neu_n in enumerate(neu):
        for index_psi, psi_n in enumerate(psi):
            C_phi = np.array([ # cordinate transfomation matrix: Z axis rotation
                [np.cos(np.radians(psi_n)), -1.0 * np.sin(np.radians(psi_n)), 0],
                [np.sin(np.radians(psi_n)), np.cos(np.radians(psi_n)), 0],
                [0, 0, 1.0]
            ])
            relative_ang_velocity = np.array([ 0, -1.0 / np.tan(np.radians(delta_g0)), 0 ])

            X_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, X_neu_theta[index_neu][index_theta])
            N_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, N_neu_theta[index_neu][index_theta])
            W_neu_theta_psi[index_neu][index_theta][index_psi] = np.cross(relative_ang_velocity, X_neu_theta_psi[index_neu][index_theta][index_psi])

            Xdash_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, Xdash_neu_theta[index_neu][index_theta])
            Ndash_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, Ndash_neu_theta[index_neu][index_theta])
            Wdash_neu_theta_psi[index_neu][index_theta][index_psi] = np.cross(relative_ang_velocity, Xdash_neu_theta_psi[index_neu][index_theta][index_psi])


'''
X_neu_psi: calculate the generating lines from the meshing conditions of gears
'''
X_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
Theta_neu_psi = np.zeros((resolution_neu, resolution_psi, 1))

for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        for index_theta, theta_n in enumerate(theta):

            if index_theta == resolution_theta -1:
                break # skip the last loop to calculate diff of elements

            dot_product1 = np.dot(N_neu_theta_psi[index_neu][index_theta][index_psi], W_neu_theta_psi[index_neu][index_theta][index_psi])
            dot_product2 = np.dot(N_neu_theta_psi[index_neu][index_theta+1][index_psi], W_neu_theta_psi[index_neu][index_theta+1][index_psi])

            if dot_product1 * dot_product2 <= 0: # if cross the zero, calculate the interpolation
                point1 = X_neu_theta_psi[index_neu][index_theta][index_psi]
                point2 = X_neu_theta_psi[index_neu][index_theta+1][index_psi]
                interpolated_vector = point1 + (abs(dot_product1)/(abs(dot_product1) + abs(dot_product2))) * (point2 - point1)
                X_neu_psi[index_neu][index_psi] = interpolated_vector
                interpolated_theta = index_theta + (abs(dot_product1)/(abs(dot_product1) + abs(dot_product2))) * (1.0)
                Theta_neu_psi[index_neu][index_psi] = interpolated_theta
                #print(index_psi, ',', index_neu, ',', interpolated_theta)
                break

Xdash_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
Theta_dash_neu_psi = np.zeros((resolution_neu, resolution_psi, 1))

for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        for index_theta, theta_n in enumerate(theta):

            if index_theta == resolution_theta -1:
                break # skip the last loop to calculate diff of elements

            dot_product1 = np.dot(Ndash_neu_theta_psi[index_neu][index_theta][index_psi], Wdash_neu_theta_psi[index_neu][index_theta][index_psi])
            dot_product2 = np.dot(Ndash_neu_theta_psi[index_neu][index_theta+1][index_psi], Wdash_neu_theta_psi[index_neu][index_theta+1][index_psi])

            if dot_product1 * dot_product2 <= 0: # if cross the zero, calculate the interpolation
                point1 = Xdash_neu_theta_psi[index_neu][index_theta][index_psi]
                point2 = Xdash_neu_theta_psi[index_neu][index_theta+1][index_psi]
                interpolated_vector = point1 + (abs(dot_product1)/(abs(dot_product1) + abs(dot_product2))) * (point2 - point1)
                Xdash_neu_psi[index_neu][index_psi] = interpolated_vector
                interpolated_theta = index_theta + (abs(dot_product1)/(abs(dot_product1) + abs(dot_product2))) * (1.0)
                Theta_dash_neu_psi[index_neu][index_psi] = interpolated_theta
                #print(index_psi, ',', index_neu, ',', interpolated_theta)
                break

'''
X_gear_neu_psi = Calculation of gear tooth surfaces through coordinate transformation of generating lines
'''
X_gear_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        if Theta_neu_psi[index_neu][index_psi] > 0:
            z_axis_rotation = -1.0 * psi_n / np.sin(np.radians(delta_g0))
            C = np.array([ # cordinate transfomation matrix: Z axis rotation
                [np.cos(np.radians(z_axis_rotation)), -1.0 * np.sin(np.radians(z_axis_rotation)), 0],
                [np.sin(np.radians(z_axis_rotation)), np.cos(np.radians(z_axis_rotation)), 0],
                [0, 0, 1.0]
            ])
            C_inv = np.linalg.inv(C)
            x_axis_rotation = 90.0 + delta_g0
            A = np.array([ # cordinate transfomation matrix: X axis rotation
                [1.0, 0, 0],
                [0, np.cos(np.radians(x_axis_rotation)), -1.0 * np.sin(np.radians(x_axis_rotation))],
                [0, np.sin(np.radians(x_axis_rotation)), np.cos(np.radians(x_axis_rotation))]
            ])
            A_inv = np.linalg.inv(A)
            X_gear_neu_psi[index_neu][index_psi] = np.matmul(C_inv, np.matmul(A_inv, X_neu_psi[index_neu][index_psi])) 

Xdash_gear_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        if Theta_dash_neu_psi[index_neu][index_psi] > 0:
            z_axis_rotation = -1.0 * psi_n / np.sin(np.radians(delta_g0))
            C = np.array([ # cordinate transfomation matrix: Z axis rotation
                [np.cos(np.radians(z_axis_rotation)), -1.0 * np.sin(np.radians(z_axis_rotation)), 0],
                [np.sin(np.radians(z_axis_rotation)), np.cos(np.radians(z_axis_rotation)), 0],
                [0, 0, 1.0]
            ])
            C_inv = np.linalg.inv(C)
            x_axis_rotation = 90.0 + delta_g0
            A = np.array([ # cordinate transfomation matrix: X axis rotation
                [1.0, 0, 0],
                [0, np.cos(np.radians(x_axis_rotation)), -1.0 * np.sin(np.radians(x_axis_rotation))],
                [0, np.sin(np.radians(x_axis_rotation)), np.cos(np.radians(x_axis_rotation))]
            ])
            A_inv = np.linalg.inv(A)
            Xdash_gear_neu_psi[index_neu][index_psi] = np.matmul(C_inv, np.matmul(A_inv, Xdash_neu_psi[index_neu][index_psi])) 

'''
#Plot X_gear_neu_psi
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        if Theta_neu_psi[index_neu][index_psi] > 0:
            color_psi = np.array([index_psi/resolution_psi, index_psi/resolution_psi, 0.5]) # RGB (0-1, 0-1, 0-1)
            ax.scatter(X_gear_neu_psi[index_neu][index_psi][0], X_gear_neu_psi[index_neu][index_psi][1], X_gear_neu_psi[index_neu][index_psi][2], c=color_psi, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.grid(True)
plt.show()
'''

'''
スプライン補間してきれいな点群にそろえる
目標： X_gear_neu_psiからgear_surface[u][v][vectors] を得ること
 - X_gear_neu_psi: ギア座標系での歯面座標
 - u: 歯たけ(profile)方向の座標, 0(根本)~1(歯先), resolutionと同じ分割量
 - v: 歯すじ(flank)方向の座標, 0(外側)~1(内側), resolutionと同じ分割量
'''
gear_surface = np.zeros((resolution_theta, resolution_neu, vector_dimention))

for index_neu, neu_n in enumerate(neu):
    thetas = np.empty(0, dtype=float)
    points = np.empty((0, 3), dtype=float)    
    for index_psi, psi_n in enumerate(psi):
        if Theta_neu_psi[index_neu][index_psi] > 0:
            thetas = np.append(thetas, Theta_neu_psi[index_neu][index_psi])
            points = np.append(points, np.array([X_gear_neu_psi[index_neu][index_psi]]), axis=0)

    thetas = np.flipud(thetas) # theta must be in ascending order.
    points = np.flipud(points) # theta must be in ascending order.

    spline_x = CubicSpline(thetas, points[:,0]) # bc_type='natural'
    spline_y = CubicSpline(thetas, points[:,1])
    spline_z = CubicSpline(thetas, points[:,2])

    for index_theta, theta_n in enumerate(theta):
        gear_surface[index_theta][index_neu] = np.array([spline_x(index_theta), spline_y(index_theta), spline_z(index_theta)])

gear_dash_surface = np.zeros((resolution_theta, resolution_neu, vector_dimention))
for index_neu, neu_n in enumerate(neu):
    thetas = np.empty(0, dtype=float)
    points = np.empty((0, 3), dtype=float)    
    for index_psi, psi_n in enumerate(psi):
        if Theta_dash_neu_psi[index_neu][index_psi] > 0:
            thetas = np.append(thetas, Theta_dash_neu_psi[index_neu][index_psi])
            points = np.append(points, np.array([Xdash_gear_neu_psi[index_neu][index_psi]]), axis=0)

    #thetas = np.flipud(thetas) # theta must be in ascending order.
    #points = np.flipud(points) # theta must be in ascending order.

    spline_x = CubicSpline(thetas, points[:,0]) # bc_type='natural'
    spline_y = CubicSpline(thetas, points[:,1])
    spline_z = CubicSpline(thetas, points[:,2])

    for index_theta, theta_n in enumerate(theta):
        gear_dash_surface[index_theta][index_neu] = np.array([spline_x(index_theta), spline_y(index_theta), spline_z(index_theta)])

'''
#plot gear_surface
'''
u = np.linspace(0.0, 1.0, resolution_theta)
v = np.linspace(0.0, 1.0, resolution_neu)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index_u, u_n in enumerate(u):
    for index_v, v_n in enumerate(v):
        ax.scatter(gear_surface[index_u][index_v][0], gear_surface[index_u][index_v][1], gear_surface[index_u][index_v][2], c='b', marker='o', s=2)
        ax.scatter(gear_dash_surface[index_u][index_v][0], gear_dash_surface[index_u][index_v][1], gear_dash_surface[index_u][index_v][2], c='b', marker='o', s=2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.grid(True)
plt.show()



