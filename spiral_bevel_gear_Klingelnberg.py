import numpy as np
#import matplotlib.pyplot as plt

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