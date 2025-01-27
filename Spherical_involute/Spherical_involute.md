# 球面インボリュートベベルギアのモデリング方法

## 参考URL
 * <a id="ref2"></a>Computerized Design of Straight Bevel Gears with Optimized Profiles for Forging, Molding, or 3D Printing [URL](https://thermalprocessing.com/computerized-design-of-straight-bevel-gears-with-optimized-profiles-for-forging-molding-or-3d-printing/)

## はじめに
* 参考URLを和訳して球面インボリュートベベルギアの3Dモデルを作りたい
* 球面インボリュートの数式は2種類の方法によって求められることが書いてある
  * 球面三角法 (Spherical trigonometry) による直接定義法(direct definition)による求め方
  * 座標変換による間接定義法(indirect definition)による求め方
* 以下では両方の導出方法を記載する.  

## Direct Definition

### 2次元の場合
* taut chord: ピンと張ったコード

<img src="図_球面インボリュート\001_2Dインボリュート.jpg" width="300">

上記の図から以下の関係式が成り立つ.

$$
tan \phi = \frac{\overline{MP}}{\overline{OM}} = \frac{\widehat{MQ}}{r_b} = \frac{r_b(\phi+\theta)}{r_b} 
$$
<div style="text-align: right;">
(EQ1)
</div>

EQ1を式変形して以下が求まる.

$$
\theta = \tan {\phi} - \phi
$$
<div style="text-align: right;">
(EQ2)
</div>

EQ2が基本的な平面インボリュートの方程式. 
* $\theta$ は極角度(polar angle)と呼ばれる. 
* $\varepsilon$ はインボリュートロール角
* $\phi$ :点Pがピッチ円上に来た時には圧力角となる

### 3次元の場合

<img src="図_球面インボリュート\002_3Dインボリュート.jpg" width="700">

* 球面インボリュートも2次元同様に $\widehat{MP}$ をほどいて行った時の点Pの軌跡(3次元カーブ)となる。
* $r_b$ が基礎円半径

$$
\widehat{MP} = r_0 \varphi = \widehat{MQ} = r_b(\phi + \theta) = r_0 \varepsilon \sin \gamma_b
$$
<div style="text-align: right;">
(EQ3)
</div>

EQ3を式変形してEQ4を得る

$$
\varphi = \varepsilon \sin \gamma_b
$$
<div style="text-align: right;">
(EQ4)
</div>

EQ4を $\theta$ について解きEQ5を得る

$$
\theta = \frac{\varphi}{\sin \gamma_b} - \phi 
$$
<div style="text-align: right;">
(EQ5)
</div>

EQ5が基本的な球面インボリュートの式となる.
△OMPに注目すると球面三角法の余弦定理より以下EQ6,7 を得る
(導出: https://hooktail.sub.jp/vectoranalysis/SphereTriangle/)

$$
\cos \gamma = \cos \varphi \cos \gamma_b + \sin \varphi \sin \gamma_b \cos 90 \degree = \cos \varphi \cos \gamma_b 
$$
<div style="text-align: right;">
(EQ6)
</div>

$$
\cos \varphi = \cos \gamma_b \cos \gamma + \sin \gamma_b \sin \gamma \cos \phi 
$$
<div style="text-align: right;">
(EQ7)
</div>

また球面三角法の正弦定理より以下EQ8を得る

$$
\frac{\sin \varphi}{\sin \phi} = \frac{\sin \gamma}{\sin 90 \degree} = \frac{\sin \gamma_b}{\sin \nu} 
$$
<div style="text-align: right;">
(EQ8)
</div>

EQ8をγについて解いてEQ9を得る

$$
\sin \gamma = \frac{\sin \varphi}{\sin \phi}
$$
<div style="text-align: right;">
(EQ9)
</div>

EQ6とEQ9をEQ7代入してcosφについてEQ10を得る.

$$
\cos \varphi = \cos \varphi {\cos \gamma_b}^2  + \frac{\sin \gamma_b \sin \varphi}{\tan \phi} 
$$
<div style="text-align: right;">
(EQ10)
</div>

EQ10をφについて解くとEQ11を得る.

$$
\tan \varphi =  \sin \gamma_b \tan \phi
$$
<div style="text-align: right;">
(EQ11)
</div>

EQ4とEQ11を使ってEQ12を得る.

$$
\tan (\varepsilon \sin \gamma_b) =  \sin \gamma_b \tan \phi
$$
<div style="text-align: right;">
(EQ12)
</div>

EQ11をEQ5に代入してEQ13を得る. 

$$
\theta = \frac{\arctan (\sin \gamma_b \tan \phi)}{\sin \gamma_b} - \phi 
$$
<div style="text-align: right;">
(EQ13)
</div>

EQ13はEQ5同様に球面インボリュート関数を表す. 


### 点Pの座標計算
以下S1座標系での点Pの成分を求める.  点Pはγの関数で与えられる.

EQ6,9を用いてEQ17を得る. 

$$
\tan \varphi = \frac{\sin \varphi}{\cos \varphi} = \frac{\sin \gamma \sin \phi \cos \gamma_b}{\cos \gamma} =  \tan \gamma \sin \phi \cos \gamma_b 
$$
<div style="text-align: right;">
(EQ17)
</div>

EQ4とEQ17から以下を得る

$$
\tan \gamma = \frac{\tan [(\theta+\phi) \sin \gamma_b]}{\sin \phi \cos \gamma_b}
$$
<div style="text-align: right;">
(EQ18)
</div>

図からS1座標系での点Pの座標は以下となる.

$$
r_1^{P} =  
\begin{bmatrix}
0 \\
r_0 \sin \gamma\\
r_0 \cos \gamma\\
1
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ19)
</div>

S0上での点Pの座表はS1をθ回転させるのでEQ20で求まる.

$$
r_0^{P} =  
\begin{bmatrix}
\cos \theta & \sin \theta & 0 & 0 \\
-\sin \theta & \cos \theta & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}^{-1} \begin{bmatrix}
0 \\
r_0 \sin \gamma\\
r_0 \cos \gamma\\
1
\end{bmatrix} = \begin{bmatrix}
-r_0 \sin \gamma \sin \theta \\
r_0 \sin \gamma \cos \theta\\
r_0 \cos \gamma\\
1
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ20)
</div>

最終的に上記のPはγbとφから求めることができる.

左歯の場合は逆向きのため以下EQ21で求まる. 

$$
r_0^{P} =  
\begin{bmatrix}
r_0 \sin \gamma \sin \theta \\
r_0 \sin \gamma \cos \theta\\
r_0 \cos \gamma\\
1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ21)
</div>

## 間接法による定義

<img src="図_球面インボリュート\003_3Dインボリュート2.jpg" width="700">

直接法では球面上を小さなディスクが転がる時にできる軌跡Pで定義したが、間接法では円錐状を大きなディスクが転がるときにできる軌跡Pとその座標変換で球面インボリュートの関数を求める.

ディスクに固定された座標系S0での点Pの座標はEQ22で与えられる.

$$
r_1^{P} =  
\begin{bmatrix}
0 \\
0\\
r_0\\
1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ22)
</div>

ここで座標変換行列を考えることによりS3座標系での点Pの座標はEQ26で与えられる.
ここでM10, M21, M32はEQ23~25の座標変換行列である.

$$
M_{10}(\varphi) = RotationCCW (y_0, \varphi) = \begin{bmatrix}
\cos \varphi & 0 & \sin \varphi & 0\\
0 & 1 & 0 & 0\\
-\sin \varphi & 0 & \cos \varphi & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ23)
</div>

$$
M_{21}(\gamma_b) = RotationCCW (x_1, \gamma_b) = \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & \cos \gamma_b & \sin \gamma_b & 0\\
0 & -\sin \gamma_b & \cos \gamma_b & 0\\
0 & 0 & 0 & 1
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ24)
</div>

$$
M_{32}(\varepsilon) = RotationCW (z_2, \varepsilon) = \begin{bmatrix}
\cos \varepsilon & \sin \varepsilon & 0 & 0\\
-\sin \varepsilon & \cos \varepsilon & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ25)
</div>

$$
r_3^{P} (\varepsilon, \varphi) =  
M_{32}(z_2,\varepsilon) M_{21}(x_1, \gamma_b) M_{10}(y_0, \varphi) r_0^{P}
= \begin{bmatrix}
r_0 (\cos \varepsilon \sin \varphi - \sin \varepsilon \cos \varphi  \sin \gamma_b)\\
r_0 (\sin \varepsilon \sin \varphi + \cos \varepsilon \cos \varphi  \sin \gamma_b)\\
r_0 (\cos \varphi \cos \gamma_b)\\
1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ26)
</div>

### 法線ベクトルの定義
間接法のメリットは法線ベクトルと正接ベクトルが簡単に導出できることである。
法線ベクトルは歯車のかみ合い解析に必要となる。

S0座標系で法線ベクトルと正接ベクトルは図を見ればEQ28, 29で与えられる。

$$
n_0^{P} =  
\begin{bmatrix}
1\\
0\\
0\\
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ28)
</div>

$$
t_0^{P} =  
\begin{bmatrix}
0\\
1\\
0\\
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ29)
</div>

よってEQ26同様に座標変換によってS3座標系での法線ベクトルと正接ベクトルがEQ30, 31で求められる.
ここでL10, L21, L32は行列Mを3x3行列にしたもの(4行列目を無視した物である)。

$$
n_3^{P} (\varepsilon, \varphi) =  
L_{32}(z_2,\varepsilon) L_{21}(x_1, \gamma_b) L_{10}(y_0, \varphi) n_0^{P}
$$
<div style="text-align: right;">
(EQ30)
</div>

$$
t_3^{P} (\varepsilon, \varphi) =  
L_{32}(z_2,\varepsilon) L_{21}(x_1, \gamma_b) L_{10}(y_0, \varphi) t_0^{P}
$$
<div style="text-align: right;">
(EQ31)
</div>

## 球面インボリュート歯面の定義
### 歯厚の定義 (Gear Tooth Thickness) 
ピッチ円状での歯厚はギア枚数Nから標準的にはEQ32で与えられる.
バックラッシュが必要な場合は適度に補正する.

$$
t_p = \frac{\pi}{N} 
$$
<div style="text-align: right;">
(EQ32)
</div>

<img src="図_球面インボリュート\004_gear_thickness.jpg" width="300">

ギアの歯の中心に沿った座標系S4に直すために以下の座標変換行列 EQ33を考える. 左歯の場合はCWにする.

$$
M_{43} = RotationCCW (z_3, \xi_p) = \begin{bmatrix}
\cos \xi_p & \sin \xi_p & 0 & 0\\
-\sin \xi_p & \cos \xi_p & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ33)
</div>

$$
M_{43} = RotationCW (z_3, \xi_p) = \begin{bmatrix}
\cos \xi_p & -\sin \xi_p & 0 & 0\\
\sin \xi_p & \cos \xi_p & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$
<div style="text-align: right;">
(EQ34)
</div>

ここでξは図よりEQ33-2で与えられる.
θpはPがピッチ円上に来た時のθである. (極角度, polar angle)

$$
\xi_p = (t_p/2) + \theta_p
$$
<div style="text-align: right;">
(EQ33-2)
</div>

### 極角度とピッチコーンの定義 (the Polar Angle at the Pitch Cone)
θpはEQ13から決めることができる. (θはγbとΦの関数)
Pがピッチ円上にある時にΦは圧力角となる

Pがピッチ円状にある時のγとΦをγpとΦpとすると, EQ11, 17からEQ37が得られえる.

$$
\sin \gamma_b \tan \phi_p = \tan \gamma_p \sin \phi_p \cos \gamma_b
$$
<div style="text-align: right;">
(EQ37)
</div>

EQ37を式変形してEQ40を得る. 

$$
\cos \phi_p = \frac{\tan \gamma_b}{ \tan \gamma_p }
$$
<div style="text-align: right;">
(EQ40)
</div>

これによりθpがEQ13からEQ41で求まる.

$$
\theta_p = \frac{\arctan (\sin \gamma_b \tan \phi_p)}{\sin \gamma_b} - \phi_p 
$$
<div style="text-align: right;">
(EQ41)
</div>

<img src="図_球面インボリュート\005_pitch_cone.jpg" width="700">

### 基礎円錐角の定義 (the Base Cone Angle)
図のO1MP0に着目して球面三角法の正弦定理よりEQ42を得る.

$$
\frac{\gamma_b}{\sin (90 \degree - \alpha)} =  \frac{\sin \gamma_p}{\sin 90 \degree} 
$$
<div style="text-align: right;">
(EQ42)
</div>

よってEQ42からEQ43が得られる。
これでピッチ円すい角から基礎円すい角が求められる. αは圧力角である.

$$
\gamma_b = \arcsin (\cos \alpha \sin \gamma_p) 
$$
<div style="text-align: right;">
(EQ43)
</div>

### 球面インボリュートベベルギアの歯面の定義
以下の図は外部基準球上に描かれた球面インボリュートプロファイルから生成された歯車歯面を示す。
球面インボリュート上の点は球の中心に向かって投影され、この方法で歯車歯面が生成される。

<img src="図_球面インボリュート\006_gear_tooth_surface.jpg" width="300">

左歯の数式はγとρを使って以下で表される。 (EQ21の左側)
ρは半径方法のパラメータ, γはEQ18によって決まるパラメータ

$$
r_0^{P} = \begin{bmatrix}
\rho_0 \sin \gamma \sin \theta \\
\rho_0 \sin \gamma \cos \theta\\
\rho_0 \cos \gamma\\
1
\end{bmatrix} 
$$
<div style="text-align: right;">
(EQ44)
</div>

ρの範囲はAo(ギア外径)からAi(ギア内径)である.
A0は以下図よりEQ45で求まる.
mはモジュール, Nは歯数, γpはピッチ角。

$$
A_o = \frac{r_p}{\sin \gamma_p} = \frac{mN}{2 \sin \gamma_p}
$$
<div style="text-align: right;">
(EQ45)
</div>

Aiは歯厚FwからEQ46で求まる. (Fwはおおよそ1/3A0程度になることが多い)

$$
A_i = A_o - F_w 
$$
<div style="text-align: right;">
(EQ46)
</div>

<img src="図_球面インボリュート\007_addendum.jpg" width="400">

### アデンダム角度とデデンダム角度の定義(Face and Root Cone Angles)
上記図のγfとγrはアデンダム係数(通常1.0)とデデンダム係数(通常1.25)によってEQ47,48で求められる.
アデンダム係数はKa, デデンダム係数はKdである.

$$
\gamma_f = \gamma_p + \arctan(\frac{k_a m}{A_o}) = \gamma_p + \arctan(\frac{2 k_a \sin \gamma_p}{N})
$$
<div style="text-align: right;">
(EQ47)
</div>

$$
\gamma_r = \gamma_p - \arctan(\frac{k_d m}{A_o}) = \gamma_p - \arctan(\frac{2 k_d \sin \gamma_p}{N})
$$
<div style="text-align: right;">
(EQ48)
</div>

## 球面インボリュートの歯面の計算
ここまでで理論が理解できたためpythonでは歯面を計算するコードを書いてみる。
半径ごとにZ軸方向に少しずつ回転させればスパイラルベベルギアにもできる
Autodesk Fusion APIを使って3Dモデルにするコードも書いたのであとはフォルダを参照

* ピニオン/ギア
  * 歯数: 11 / 23
  * 歯幅: 25mm / 25mm
  * 歯直角モジュール: 5
  * スパイラル角: 32°
  * 圧力角: 20°
  * スパイラル円弧径: 300mm

```python
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.interpolate import CubicSpline
import matplotlib.tri as tri
from pyscript import display

model = "pinion" # "gear" or "pinion"
N_p = 23 # num of tooth pinion
N_g = 11 # num of tooth gear
module = 5 # ギアミルで確認 歯直角モジュール
Fw = 25 # mm /Face width
Sigma = 90 # deg / shaft angle
alpha = 20 # deg / pressure angle
ka = 1.0 # addendam coefficient
kd = 1.25 # dedendam coefficient
beta = 32 # deg / spral angle
rc = 300 # mm / cutter radius

i = N_g / N_p # Reduction ratio
rp_g = np.degrees(np.arctan2(np.sin(np.radians(Sigma)), 1/i + np.cos(np.radians(Sigma)))) # deg / pitch angle of gear
rp_p = Sigma - rp_g # deg / pitch angle of pinion

#EQ45, EQ46
Ao = (module * N_g) / (2 * np.sin(np.radians(rp_g)))
Ai = Ao - Fw
Am = (Ao + Ai)/2

#EQ47, EQ48, EQ43
#gamma_f = rp_g + np.degrees(np.arctan(2 * ka * np.sin(np.radians(rp_g))/N_g))
gamma_f = rp_g + np.degrees(np.arctan(ka * module / Ao))
#gamma_r = rp_g - np.degrees(np.arctan(2 * kd * np.sin(np.radians(rp_g))/N_g))
gamma_r = rp_g - np.degrees(np.arctan(kd * module / Ao))
gamma_b = np.degrees(np.arcsin(np.cos(np.radians(alpha)) * np.sin(np.radians(rp_g))))

#EQ32, EQ41, EQ33-2
t_p = np.degrees(np.pi) / N_g # deg ピッチ円上の歯厚角度
cos_phi_p = np.tan(np.radians(gamma_b)) / np.tan(np.radians(rp_g))
phi_p = np.degrees(np.arccos(cos_phi_p))
theta_p = np.degrees(np.arctan(np.sin(np.radians(gamma_b)) * np.tan(np.radians(phi_p)))) / (np.sin(np.radians(gamma_b))) - phi_p # deg ピッチ円上のθ
xi_p = (t_p/2) + theta_p # deg 歯の中心とのずれ角度

# スパイラル部分のための計算
rcl = np.sqrt(Am*Am + rc*rc - 2* Am * rc * np.cos(np.radians(90.0 - beta))) # カッター位置 (余弦定理) 
cos_d0 = (Am*Am + rcl*rcl - rc*rc) / (2 * Am * rcl) # Amでの角度d (余弦定理) 
d0 = np.degrees(np.arccos(cos_d0))

print("rp_g", rp_g)
print("Ao", Ao)
print("Ai", Ai)
print("gamma_f", gamma_f)
print("gamma_r", gamma_r)
print("gamma_b", gamma_b)
print("t_p", t_p)
print("theta_p", theta_p)
print("xi_p", xi_p)
print("rcl", rcl)
print("d0", d0)

# 変数類の用意
rho_gear_surface = np.linspace(Ai, Ao, 20) # mm
rho_root_surface = np.linspace(Ai, Ao, 20) # mm

if gamma_b > gamma_r: # 基礎円が歯元より低い場合 (ピニオン等)
  gamma_gear_surface = np.linspace(gamma_b, gamma_f, 20) # deg
  gamma_root_surface = np.linspace(gamma_r, gamma_b, 20) # deg
else:
  gamma_gear_surface = np.linspace(gamma_r, gamma_f, 20) # deg
  gamma_root_surface = np.linspace(gamma_b, gamma_r, 20) # deg

gear_surface_right = np.zeros((20, 20, 3))
gear_surface_left = np.zeros((20, 20, 3))
root_surface_right = np.zeros((20, 20, 3))
root_surface_left = np.zeros((20, 20, 3))

# 歯面計算
for index_gamma, gamma_n in enumerate(gamma_gear_surface):
    for index_rho, rho_n in enumerate(rho_gear_surface):

        #decide varphi EQ16
        cos_varphi = np.cos(np.radians(gamma_n)) / np.cos(np.radians(gamma_b))
        varphi = np.degrees(np.arccos(cos_varphi))

        #decide phi EQ11
        tan_phi = np.tan(np.radians(varphi)) / np.sin(np.radians(gamma_b))
        phi = np.degrees(np.arctan2(tan_phi,1))

        #decide theta EQ13
        theta = (np.degrees(np.arctan(np.sin(np.radians(gamma_b)) * np.tan(np.radians(phi)))) / (np.sin(np.radians(gamma_b)))) - phi

        #calicurate spherical involute: right surface EQ20
        X_right = np.array([
            -1 * rho_n * np.sin(np.radians(gamma_n)) * np.sin(np.radians(theta)),
            +1 * rho_n * np.sin(np.radians(gamma_n)) * np.cos(np.radians(theta)),
            rho_n * np.cos(np.radians(gamma_n))
        ])

        #calicurate spherical involute: left surface EQ21
        X_left = np.array([
            +1 * rho_n * np.sin(np.radians(gamma_n)) * np.sin(np.radians(theta)),
            +1 * rho_n * np.sin(np.radians(gamma_n)) * np.cos(np.radians(theta)),
            rho_n * np.cos(np.radians(gamma_n))
        ])

        #calicurate rotation matrix: right side EQ33
        RotationCCW_M43 = np.array([
            [+1 * np.cos(np.radians(xi_p)),np.sin(np.radians(xi_p)),0],
            [-1 * np.sin(np.radians(xi_p)),np.cos(np.radians(xi_p)),0],
            [0,0,1]
        ])

        #calicurate rotation matrix: left side EQ34
        RotationCW_M43 = np.array([
            [np.cos(np.radians(xi_p)),-1 * np.sin(np.radians(xi_p)),0],
            [np.sin(np.radians(xi_p)),+1 * np.cos(np.radians(xi_p)),0],
            [0,0,1]
        ])

        #gear surface calicurarion
        gear_surface_right[index_gamma][index_rho] = np.matmul(RotationCCW_M43,X_right)
        gear_surface_left[index_gamma][index_rho] = np.matmul(RotationCW_M43,X_left)

        #spiral部分の回転計算
        cos_d = (rho_n*rho_n + rcl*rcl - rc*rc) / (2 * rho_n * rcl) # rho_nでの角度d (余弦定理) 
        d = np.degrees(np.arccos(cos_d))
        if model == "pinion":
            d = d * N_p / N_g

        #calicurate rotation matrix: d
        RotationCW_M43 = np.array([
            [np.cos(np.radians(d - d0)),-1 * np.sin(np.radians(d - d0)),0],
            [np.sin(np.radians(d - d0)),+1 * np.cos(np.radians(d - d0)),0],
            [0,0,1]
        ])
        gear_surface_right[index_gamma][index_rho] = np.matmul(RotationCW_M43,gear_surface_right[index_gamma][index_rho])
        gear_surface_left[index_gamma][index_rho] = np.matmul(RotationCW_M43,gear_surface_left[index_gamma][index_rho])

        #print(gear_surface_right[index_gamma][index_rho][0],",", gear_surface_right[index_gamma][index_rho][1],",", gear_surface_right[index_gamma][index_rho][2])
        #print(gear_surface_left[index_gamma][index_rho][0],",", gear_surface_left[index_gamma][index_rho][1],",", gear_surface_left[index_gamma][index_rho][2])

# 歯元面計算
if gamma_b > gamma_r:
    for index_gamma, gamma_n in enumerate(gamma_root_surface):
        for index_rho, rho_n in enumerate(rho_root_surface):
            #root surface right
            X_right = np.array([
                0,
                rho_n * np.sin(np.radians(gamma_n)),
                rho_n * np.cos(np.radians(gamma_n))
            ])

            #root surface left
            X_left = np.array([
                0,
                rho_n * np.sin(np.radians(gamma_n)),
                rho_n * np.cos(np.radians(gamma_n))
            ])

            #calicurate rotation matrix: right side EQ33
            RotationCCW_M43 = np.array([
                [+1 * np.cos(np.radians(xi_p)),np.sin(np.radians(xi_p)),0],
                [-1 * np.sin(np.radians(xi_p)),np.cos(np.radians(xi_p)),0],
                [0,0,1]
            ])

            #calicurate rotation matrix: left side EQ34
            RotationCW_M43 = np.array([
                [np.cos(np.radians(xi_p)),-1 * np.sin(np.radians(xi_p)),0],
                [np.sin(np.radians(xi_p)),+1 * np.cos(np.radians(xi_p)),0],
                [0,0,1]
            ])

            root_surface_right[index_gamma][index_rho] = np.matmul(RotationCCW_M43,X_right)
            root_surface_left[index_gamma][index_rho] = np.matmul(RotationCW_M43,X_left)

            #spiral部分の回転計算
            cos_d = (rho_n*rho_n + rcl*rcl - rc*rc) / (2 * rho_n * rcl) # rho_nでの角度d (余弦定理)
            d = np.degrees(np.arccos(cos_d))
            if model == "pinion":
                d = d * N_p / N_g

            #calicurate rotation matrix: d
            RotationCW_M43 = np.array([
                [np.cos(np.radians(d - d0)),-1 * np.sin(np.radians(d - d0)),0],
                [np.sin(np.radians(d - d0)),+1 * np.cos(np.radians(d - d0)),0],
                [0,0,1]
            ])

            root_surface_right[index_gamma][index_rho] = np.matmul(RotationCW_M43,root_surface_right[index_gamma][index_rho])
            root_surface_left[index_gamma][index_rho] = np.matmul(RotationCW_M43,root_surface_left[index_gamma][index_rho])

            #print(root_surface_right[index_gamma][index_rho][0],",", root_surface_right[index_gamma][index_rho][1],",", root_surface_right[index_gamma][index_rho][2])
            #print(root_surface_left[index_gamma][index_rho][0],",", root_surface_left[index_gamma][index_rho][1],",", root_surface_left[index_gamma][index_rho][2])

# visualization of gear surface
def get_point_cloud():
    point_cloud = []
    for index_gamma, gamma_n in enumerate(gamma_gear_surface):
        for index_rho, rho_n in enumerate(rho_gear_surface):
            point_cloud.append([gear_surface_right[index_gamma][index_rho][0], gear_surface_right[index_gamma][index_rho][1], gear_surface_right[index_gamma][index_rho][2]])
            point_cloud.append([gear_surface_left[index_gamma][index_rho][0], gear_surface_left[index_gamma][index_rho][1], gear_surface_left[index_gamma][index_rho][2]])

    for index_gamma, gamma_n in enumerate(gamma_root_surface):
        for index_rho, rho_n in enumerate(rho_root_surface):
            point_cloud.append([root_surface_right[index_gamma][index_rho][0], root_surface_right[index_gamma][index_rho][1], root_surface_right[index_gamma][index_rho][2]])
            point_cloud.append([root_surface_left[index_gamma][index_rho][0], root_surface_left[index_gamma][index_rho][1], root_surface_left[index_gamma][index_rho][2]])

    # 残りの歯面も描画
    for index_gamma, gamma_n in enumerate(gamma_gear_surface):
        for index_rho, rho_n in enumerate(rho_gear_surface):
            for i in range(1, N_g, 1):
                RotationCCW_M43 = np.array([
                    [+1 * np.cos(np.radians(t_p*2)),np.sin(np.radians(t_p*2)),0],
                    [-1 * np.sin(np.radians(t_p*2)),np.cos(np.radians(t_p*2)),0],
                    [0,0,1]
                ])
                gear_surface_right[index_gamma][index_rho] = np.matmul(RotationCCW_M43,gear_surface_right[index_gamma][index_rho])
                gear_surface_left[index_gamma][index_rho] = np.matmul(RotationCCW_M43,gear_surface_left[index_gamma][index_rho])
                root_surface_right[index_gamma][index_rho] = np.matmul(RotationCCW_M43,root_surface_right[index_gamma][index_rho])
                root_surface_left[index_gamma][index_rho] = np.matmul(RotationCCW_M43,root_surface_left[index_gamma][index_rho])
                point_cloud.append([gear_surface_right[index_gamma][index_rho][0], gear_surface_right[index_gamma][index_rho][1], gear_surface_right[index_gamma][index_rho][2]])
                point_cloud.append([gear_surface_left[index_gamma][index_rho][0], gear_surface_left[index_gamma][index_rho][1], gear_surface_left[index_gamma][index_rho][2]])
                point_cloud.append([root_surface_right[index_gamma][index_rho][0], root_surface_right[index_gamma][index_rho][1], root_surface_right[index_gamma][index_rho][2]])
                point_cloud.append([root_surface_left[index_gamma][index_rho][0], root_surface_left[index_gamma][index_rho][1], root_surface_left[index_gamma][index_rho][2]])
    return np.array(point_cloud)

test_data = get_point_cloud()
fig = plt.figure(figsize = (8, 8))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(test_data[:,0],test_data[:,1],test_data[:,2], s = 1, c = "blue")
ax.view_init(elev=40, azim=90) #グラフの画角調整
        
# 参考用の球の作成(半径Ao)
phi = np.linspace(0, np.pi, 100)   # 仰角 (0からπ)
theta = np.linspace(0, 2 * np.pi, 100)  # 方位角 (0から2π)
phi, theta = np.meshgrid(phi, theta)# メッシュグリッドを生成
## 球の座標を計算
x = Ao * np.sin(phi) * np.cos(theta)
y = Ao * np.sin(phi) * np.sin(theta)
z = Ao * np.cos(phi)
ax.plot_surface(x, y, z, color='b', alpha=0.3)  # alpha=0.3で半透明

#参考用の円を作成(rpの位置)
r = Ao * np.sin(np.radians(rp_g))  # 半径
t = np.linspace(0, 2 * np.pi, 100)  # tは0から2πまで
t2 =  np.linspace(1, 1, 100)
## XY平面での円 (z = 0)
x = r * np.cos(t)
y = r * np.sin(t)
z = Ao * np.cos(np.radians(rp_g)) * t2  # z座標は0で固定
ax.plot(x, y, z, color='r')# 円をプロット

#plt.show()
display(fig, target="mpl")
```