# 球面インボリュートのモデリング検討

## 参考URL
 * <a id="ref2"></a>Computerized Design of Straight Bevel Gears with Optimized Profiles for Forging, Molding, or 3D Printing [URL](https://thermalprocessing.com/computerized-design-of-straight-bevel-gears-with-optimized-profiles-for-forging-molding-or-3d-printing/)

## 序章
* 球面インボリュートの歯車は鍛造や3Dプリントによって製造されるだろう。
* 球面インボリュートの数式は2種類の方法によって求められる
  * 球面三角法 (Spherical trigonometry) による直接定義法(direct definition)による求め方
  * 座標変換による間接定義法(indirect definition)による求め方

## Direct Definition

### 2次元の場合
* taut chord: ピンと張ったコード

<img src="図_球面インボリュート\001_2Dインボリュート.jpg" width="300">

上記の図から以下の関係式が成り立つ.

$$
\tan \phi = \frac{\overline{MP}}{\overline{OM}} = \frac{\widehat{MQ}}{r_b} = \frac{r_b(\phi+\theta)}{r_b} \tag{EQ1}
$$

EQ1を式変形して以下が求まる.

$$
\theta = \tan {\phi} - \phi \tag{EQ2}
$$

EQ2が基本的な平面インボリュートの方程式. 
* $\theta$ は極角度(polar angle)と呼ばれる. 
* $\varepsilon$ はインボリュートロール角
* $\phi$ :点Pがピッチ円上に来た時には圧力角となる

### 3次元の場合

<img src="図_球面インボリュート\002_3Dインボリュート.jpg" width="700">

* 球面インボリュートも2次元同様に $\widehat{MP}$ をほどいて行った時の点Pの軌跡(3次元カーブ)となる。
* $r_b$ が基礎円半径

$$
\widehat{MP} = r_0 \varphi = \widehat{MQ} = r_b(\phi + \theta) = r_0 \varepsilon \sin \gamma_b \tag{EQ3}
$$

EQ3を式変形してEQ4を得る

$$
\varphi = \varepsilon \sin \gamma_b \tag{EQ4}
$$

EQ4を $\theta$ について解きEQ5を得る

$$
\theta = \frac{\varphi}{\sin \gamma_b} - \phi  \tag{EQ5}
$$

EQ5が基本的な球面インボリュートの式となる.
△OMPに注目すると球面三角法の余弦定理より以下EQ6,7 を得る
(導出: https://hooktail.sub.jp/vectoranalysis/SphereTriangle/)

$$
\cos \gamma = \cos \varphi \cos \gamma_b + \sin \varphi \sin \gamma_b \cos 90 \degree = \cos \varphi \cos \gamma_b  \tag{EQ6}
$$

$$
\cos \varphi = \cos \gamma_b \cos \gamma + \sin \gamma_b \sin \gamma \cos \phi   \tag{EQ7}
$$

また球面三角法の正弦定理より以下EQ8を得る

$$
\frac{\sin \varphi}{\sin \phi} = \frac{\sin \gamma}{\sin 90 \degree} = \frac{\sin \gamma_b}{\sin \nu}  \tag{EQ8}
$$

EQ8をγについて解いてEQ9を得る

$$
\sin \gamma = \frac{\sin \varphi}{\sin \phi}  \tag{EQ9}
$$

EQ6とEQ9をEQ7代入してcosφについてEQ10を得る.

$$
\cos \varphi = \cos \varphi {\cos \gamma_b}^2  + \frac{\sin \gamma_b \sin \varphi}{\tan \phi}  \tag{EQ10}
$$

EQ10をφについて解くとEQ11を得る.

$$
\tan \varphi =  \sin \gamma_b \tan \phi \tag{EQ11}
$$

EQ4とEQ11を使ってEQ12を得る.

$$
\tan (\varepsilon \sin \gamma_b) =  \sin \gamma_b \tan \phi \tag{EQ12}
$$

EQ11をEQ5に代入してEQ13を得る. 

$$
\theta = \frac{\arctan (\sin \gamma_b \tan \phi)}{\sin \gamma_b} - \phi  \tag{EQ13}
$$

EQ13はEQ5同様に球面インボリュート関数を表す. 


### 点Pの座標計算
以下S1座標系での点Pの成分を求める.  点Pはγの関数で与えられる.

EQ6,9を用いてEQ17を得る. 

$$
\tan \varphi = \frac{\sin \varphi}{\cos \varphi} = \frac{\sin \gamma \sin \phi \cos \gamma_b}{\cos \gamma} =  \tan \gamma \sin \phi \cos \gamma_b \tag{EQ17}
$$

EQ4とEQ17から以下を得る

$$
\tan \gamma = \frac{\tan [(\theta+\phi) \sin \gamma_b]}{\sin \phi \cos \gamma_b} \tag{EQ18}
$$

図からS1座標系での点Pの座標は以下となる.

$$
r_1^{P} =  
\begin{bmatrix}
0 \\
r_0 \sin \gamma\\
r_0 \cos \gamma\\
1
\end{bmatrix} \tag{EQ19}
$$

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
\end{bmatrix} \tag{EQ20}
$$

最終的に上記のPはγbとφから求めることができる.

左歯の場合は逆向きのため以下EQ21で求まる. 

$$
r_0^{P} =  
\begin{bmatrix}
r_0 \sin \gamma \sin \theta \\
r_0 \sin \gamma \cos \theta\\
r_0 \cos \gamma\\
1
\end{bmatrix} \tag{EQ21}
$$

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
\end{bmatrix} \tag{EQ22}
$$

ここで座標変換行列を考えることによりS3座標系での点Pの座標はEQ26で与えられる.
ここでM10, M21, M32はEQ23~25の座標変換行列である.

$$
M_{10}(\varphi) = RotationCCW (y_0, \varphi) = \begin{bmatrix}
\cos \varphi & 0 & \sin \varphi & 0\\
0 & 1 & 0 & 0\\
-\sin \varphi & 0 & \cos \varphi & 0\\
0 & 0 & 0 & 1
\end{bmatrix} \tag{EQ23}
$$

$$
M_{21}(\gamma_b) = RotationCCW (x_1, \gamma_b) = \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & \cos \gamma_b & \sin \gamma_b & 0\\
0 & -\sin \gamma_b & \cos \gamma_b & 0\\
0 & 0 & 0 & 1
\end{bmatrix} \tag{EQ24}
$$

$$
M_{32}(\varepsilon) = RotationCW (z_2, \varepsilon) = \begin{bmatrix}
\cos \varepsilon & \sin \varepsilon & 0 & 0\\
-\sin \varepsilon & \cos \varepsilon & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix} \tag{EQ25}
$$

$$
r_3^{P} (\varepsilon, \varphi) =  
M_{32}(z_2,\varepsilon) M_{21}(x_1, \gamma_b) M_{10}(y_0, \varphi) r_0^{P}
= \begin{bmatrix}
r_0 (\cos \varepsilon \sin \varphi - \sin \varepsilon \cos \varphi  \sin \gamma_b)\\
r_0 (\sin \varepsilon \sin \varphi + \cos \varepsilon \cos \varphi  \sin \gamma_b)\\
r_0 (\cos \varphi \cos \gamma_b)\\
1
\end{bmatrix} \tag{EQ26}
$$

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
\end{bmatrix} \tag{EQ28}
$$

$$
t_0^{P} =  
\begin{bmatrix}
0\\
1\\
0\\
\end{bmatrix} \tag{EQ29}
$$

よってEQ26同様に座標変換によってS3座標系での法線ベクトルと正接ベクトルがEQ30, 31で求められる.
ここでL10, L21, L32は行列Mを3x3行列にしたもの(4行列目を無視した物である)。

$$
n_3^{P} (\varepsilon, \varphi) =  
L_{32}(z_2,\varepsilon) L_{21}(x_1, \gamma_b) L_{10}(y_0, \varphi) n_0^{P} \tag{EQ30}
$$

$$
t_3^{P} (\varepsilon, \varphi) =  
L_{32}(z_2,\varepsilon) L_{21}(x_1, \gamma_b) L_{10}(y_0, \varphi) t_0^{P} \tag{EQ31}
$$

## 球面インボリュート歯面の定義
### 歯厚の定義 (Gear Tooth Thickness) 
ピッチ円状での歯厚はギア枚数Nから標準的にはEQ32で与えられる.
バックラッシュが必要な場合は適度に補正する.

$$
t_p = \frac{\pi}{N} \tag{EQ32}
$$

<img src="図_球面インボリュート\004_gear_thickness.jpg" width="300">

ギアの歯の中心に沿った座標系S4に直すために以下の座標変換行列 EQ33を考える. 左歯の場合はCWにする.

$$
M_{43} = RotationCCW (z_3, \xi_p) = \begin{bmatrix}
\cos \xi_p & \sin \xi_p & 0 & 0\\
-\sin \xi_p & \cos \xi_p & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix} \tag{EQ33}
$$

$$
M_{43} = RotationCW (z_3, \xi_p) = \begin{bmatrix}
\cos \xi_p & -\sin \xi_p & 0 & 0\\
\sin \xi_p & \cos \xi_p & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix} \tag{EQ34}
$$

ここでξは図よりEQ33-2で与えられる.
θpはPがピッチ円上に来た時のθである. (極角度, polar angle)

$$
\xi_p = (t_p/2) + \theta_p \tag{EQ33-2}
$$

### 極角度とピッチコーンの定義 (the Polar Angle at the Pitch Cone)
θpはEQ13から決めることができる. (θはγbとΦの関数)
Pがピッチ円上にある時にΦは圧力角となる

Pがピッチ円状にある時のγとΦをγpとΦpとすると, EQ11, 17からEQ37が得られえる.

$$
\sin \gamma_b \tan \phi_p = \tan \gamma_p \sin \phi_p \cos \gamma_b \tag{EQ37}
$$

EQ37を式変形してEQ40を得る. 

$$
\cos \phi_p = \frac{\tan \gamma_b}{ \tan \gamma_p } \tag{EQ40}
$$

これによりθpがEQ13からEQ41で求まる.

$$
\theta_p = \frac{\arctan (\sin \gamma_b \tan \phi_p)}{\sin \gamma_b} - \phi_p  \tag{EQ41}
$$

<img src="図_球面インボリュート\005_pitch_cone.jpg" width="700">

### 基礎円錐角の定義 (the Base Cone Angle)
図のO1MP0に着目して球面三角法の正弦定理よりEQ42を得る.

$$
\frac{\gamma_b}{\sin (90 \degree - \alpha)} =  \frac{\sin \gamma_p}{\sin 90 \degree} \tag{EQ42}
$$

よってEQ42からEQ43が得られる。
これでピッチ円すい角から基礎円すい角が求められる. αは圧力角である.

$$
\gamma_b = \arcsin (\cos \alpha \sin \gamma_p) \tag{EQ43}
$$

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
\end{bmatrix} \tag{EQ44}
$$

ρの範囲はAo(ギア外径)からAi(ギア内径)である.
A0は以下図よりEQ45で求まる.
mはモジュール, Nは歯数, γpはピッチ角。

$$
A_o = \frac{r_p}{\sin \gamma_p} = \frac{mN}{2 \sin \gamma_p} \tag{EQ45}
$$

Aiは歯厚FwからEQ46で求まる. (Fwはおおよそ1/3A0程度になることが多い)

$$
A_i = A_o - F_w \tag{EQ46}
$$

<img src="図_球面インボリュート\007_addendum.jpg" width="400">

### アデンダム角度とデデンダム角度の定義(Face and Root Cone Angles)
上記図のγfとγrはアデンダム係数(通常1.0)とデデンダム係数(通常1.25)によってEQ47,48で求められる.
アデンダム係数はKa, デデンダム係数はKdである.

$$
\gamma_f = \gamma_p + \arctan(\frac{k_a m}{A_o}) = \gamma_p + \arctan(\frac{2 k_a \sin \gamma_p}{N}) \tag{EQ47}
$$

$$
\gamma_r = \gamma_p - \arctan(\frac{k_d m}{A_o}) = \gamma_p - \arctan(\frac{2 k_d \sin \gamma_p}{N}) \tag{EQ48}
$$

## ベベルギアの諸元
* ピニオン/ギア
  * 歯数: 16 / 40
  * 歯幅: 185mm / 185mm
  * ピッチ円直径: 540mm / 1350mm
  * スパイラル角: 32°
  * 圧力角: 20°