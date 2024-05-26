import numpy as np
import matplotlib.pyplot as plt

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
Exb = 4.5 # mm / radious difference
Z0 = 5 # num of thread of cutter
module = 24.799

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
neu = np.linspace(neu_range_neg, neu_range_pos, resolution_neu) # deg

'''
X_neu_theta: Coordinate values of the imaginary crown gear tooth surface point cloud
'''
theta = np.linspace(-1.0 * (1.25 * module)/(np.cos(np.radians(alpha))), (1.0 * module)/(np.cos(np.radians(alpha))), resolution_theta) # mm
psi = np.linspace(-15, 15, resolution_psi) # deg

X_neu_theta = np.zeros((resolution_neu, resolution_theta, vector_dimention))
for index_theta, theta_n in enumerate(theta):
    for index_neu, neu_n in enumerate(neu):
        Xc_theta = np.array([
            theta_n * 0, 
            theta_n * np.sin(np.radians(alpha)) + rc, 
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

'''
X_neu_theta_psi: imaginary crown gear coordinates in absolute coordinate system
'''
X_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
N_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))
W_neu_theta_psi = np.zeros((resolution_neu, resolution_theta, resolution_psi, vector_dimention))

for index_theta, theta_n in enumerate(theta):
    for index_neu, neu_n in enumerate(neu):
        for index_psi, psi_n in enumerate(psi):
            C_phi = np.array([ # cordinate transfomation matrix: Z axis rotation
                [np.cos(np.radians(psi_n)), -1.0 * np.sin(np.radians(psi_n)), 0],
                [np.sin(np.radians(psi_n)), np.cos(np.radians(psi_n)), 0],
                [0, 0, 1.0]
            ])
            X_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, X_neu_theta[index_neu][index_theta])
            N_neu_theta_psi[index_neu][index_theta][index_psi] = np.matmul(C_phi, N_neu_theta[index_neu][index_theta])

            relative_ang_velocity = np.array([
                0, -1.0 / np.tan(np.radians(delta_g0)), 0
            ])
            W_neu_theta_psi[index_neu][index_theta][index_psi] = np.cross(relative_ang_velocity, X_neu_theta_psi[index_neu][index_theta][index_psi])

'''
X_neu_psi: calculate the generating lines from the meshing conditions of gears
'''
X_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
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
                print(index_psi, ',', index_neu, ',', interpolated_theta)
                break

'''
X_gear_neu_psi = Calculation of gear tooth surfaces through coordinate transformation of generating lines
'''
X_gear_neu_psi = np.zeros((resolution_neu, resolution_psi, vector_dimention))
for index_psi, psi_n in enumerate(psi):
    for index_neu, neu_n in enumerate(neu):
        if X_neu_psi[index_neu][index_psi] is not None:
            break
