import adsk.core, adsk.fusion, traceback
import numpy as np
from scipy.interpolate import CubicSpline

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface   
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)# Create a document.
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent

        # サーフェス点群の定義
        u = np.linspace(0,10,50)
        v = np.linspace(0,10,50)
        surface_points = np.zeros((50, 50, 3))
        for index_u, u_n in enumerate(u):
            for index_v, v_n in enumerate(v):
                surface_points[index_u][index_v] = np.array([u_n, v_n, 2*np.cos(u_n) + 3*np.sin(v_n)])


        #Create loft feature input
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        oftSectionsObj = loftInput.loftSections
        loftInput.isSolid = False
        loftInput.isClosed = False
        loftInput.isTangentEdgesMerged = True

        #ロフトに点群を入れ込む
        for index_u, u_n in enumerate(u):
            sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
            points = adsk.core.ObjectCollection.create()
            for index_v, v_n in enumerate(v):
                points.add(adsk.core.Point3D.create(surface_points[index_u][index_v][0], surface_points[index_u][index_v][1], surface_points[index_u][index_v][2]))
            curve = sketch.sketchCurves.sketchFittedSplines.add(points)
            section = rootComp.features.createPath(curve)
            loftInput.loftSections.add(section)

        #ロフト生成
        loftFeats.add(loftInput)

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))