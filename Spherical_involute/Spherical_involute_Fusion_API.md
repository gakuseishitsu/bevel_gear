#Author-
#Description-

import adsk.core, adsk.fusion, adsk.cam, traceback
import numpy as np
from scipy.interpolate import CubicSpline

def run(context):
    ui = None
    try:

        model = "pinion" # "gear" or "pinion"
        N_p = 57 # num of tooth pinion
        N_g = 11 # num of tooth gear
        module = 2 # ギアミルで確認 歯直角モジュール
        Fw = 15 # mm /Face width
        Sigma = 90 # deg / shaft angle
        alpha = 20 # deg / pressure angle
        ka = 1.0 # addendam coefficient
        kd = 1.25 # dedendam coefficient
        beta = 32 # deg / spral angle
        rc = 100 # mm / cutter radius

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


        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('SBG_spherical_involute_gear')
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)# Create a document.
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent


        '''
        右歯面の作成
        '''
        #Create loft feature input
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        oftSectionsObj = loftInput.loftSections
        loftInput.isSolid = False
        loftInput.isClosed = False
        loftInput.isTangentEdgesMerged = True

        for index_rho, rho_n in enumerate(rho_gear_surface):
            sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
            points = adsk.core.ObjectCollection.create()
            for index_gamma, gamma_n in enumerate(gamma_gear_surface):
                points.add(adsk.core.Point3D.create(gear_surface_right[index_gamma][index_rho][0], gear_surface_right[index_gamma][index_rho][1], gear_surface_right[index_gamma][index_rho][2]))
            curve = sketch.sketchCurves.sketchFittedSplines.add(points)
            section = rootComp.features.createPath(curve)
            loftInput.loftSections.add(section)

        #ロフト生成
        loftFeats.add(loftInput)

        '''
        左歯面の作成
        '''
        #Create loft feature input
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        oftSectionsObj = loftInput.loftSections
        loftInput.isSolid = False
        loftInput.isClosed = False
        loftInput.isTangentEdgesMerged = True

        for index_rho, rho_n in enumerate(rho_gear_surface):
            sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
            points = adsk.core.ObjectCollection.create()
            for index_gamma, gamma_n in enumerate(gamma_gear_surface):
                points.add(adsk.core.Point3D.create(gear_surface_left[index_gamma][index_rho][0], gear_surface_left[index_gamma][index_rho][1], gear_surface_left[index_gamma][index_rho][2]))
            curve = sketch.sketchCurves.sketchFittedSplines.add(points)
            section = rootComp.features.createPath(curve)
            loftInput.loftSections.add(section)

        #ロフト生成
        loftFeats.add(loftInput)


        '''
        右歯底面の作成
        '''
        if gamma_b > gamma_r:
            #Create loft feature input
            loftFeats = rootComp.features.loftFeatures
            loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            oftSectionsObj = loftInput.loftSections
            loftInput.isSolid = False
            loftInput.isClosed = False
            loftInput.isTangentEdgesMerged = True

            for index_rho, rho_n in enumerate(rho_gear_surface):
                sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
                points = adsk.core.ObjectCollection.create()
                for index_gamma, gamma_n in enumerate(gamma_gear_surface):
                    points.add(adsk.core.Point3D.create(root_surface_right[index_gamma][index_rho][0], root_surface_right[index_gamma][index_rho][1], root_surface_right[index_gamma][index_rho][2]))
                curve = sketch.sketchCurves.sketchFittedSplines.add(points)
                section = rootComp.features.createPath(curve)
                loftInput.loftSections.add(section)

            #ロフト生成
            loftFeats.add(loftInput)

        '''
        左歯底面の作成
        '''
        if gamma_b > gamma_r:
            #Create loft feature input
            loftFeats = rootComp.features.loftFeatures
            loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            oftSectionsObj = loftInput.loftSections
            loftInput.isSolid = False
            loftInput.isClosed = False
            loftInput.isTangentEdgesMerged = True

            for index_rho, rho_n in enumerate(rho_gear_surface):
                sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
                points = adsk.core.ObjectCollection.create()
                for index_gamma, gamma_n in enumerate(gamma_gear_surface):
                    points.add(adsk.core.Point3D.create(root_surface_left[index_gamma][index_rho][0], root_surface_left[index_gamma][index_rho][1], root_surface_left[index_gamma][index_rho][2]))
                curve = sketch.sketchCurves.sketchFittedSplines.add(points)
                section = rootComp.features.createPath(curve)
                loftInput.loftSections.add(section)

            #ロフト生成
            loftFeats.add(loftInput)

        '''
        基準円や線の作図
        '''
        sketches = rootComp.sketches;
        xyPlane = rootComp.yZConstructionPlane
        sketch = sketches.add(xyPlane)
        circles = sketch.sketchCurves.sketchCircles
        circle1 = circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), Ao)
        circle1 = circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), Ai)
        lines = sketch.sketchCurves.sketchLines;
        line1 = lines.addByTwoPoints(adsk.core.Point3D.create(0, 0, 0), adsk.core.Point3D.create(-1*Ao * np.cos(np.radians(gamma_r)), Ao * np.sin(np.radians(gamma_r)), 0)) #z,y,x
        line1 = lines.addByTwoPoints(adsk.core.Point3D.create(0, 0, 0), adsk.core.Point3D.create(-1*Ao * np.cos(np.radians(gamma_f)), Ao * np.sin(np.radians(gamma_f)), 0)) #z,y,x

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
