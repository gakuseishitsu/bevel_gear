import adsk.core, adsk.fusion, traceback

def create_loft_profile(profile_points, root_comp, loft_features):
    sketch = root_comp.sketches.add(root_comp.xZConstructionPlane)
    points = adsk.core.ObjectCollection.create()
    for point in profile_points:
        points.add(adsk.core.Point3D.create(*point))
    curve = sketch.sketchCurves.sketchFittedSplines.add(points)
    return root_comp.features.createPath(curve)

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface   
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)# Create a document.
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent

        profiles = [
            [(0, 0, 0), (5, 1, 0), (6, 4, 3)],
            [(0, 0, 1), (5, 1, 1), (6, 4, 4)],
            [(0, 0, 2), (5, 1, 2), (6, 4, 5)]
        ]

        ### Create loft feature input
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        oftSectionsObj = loftInput.loftSections
        loftInput.isSolid = False
        loftInput.isClosed = False
        loftInput.isTangentEdgesMerged = True

        for profile_points in profiles:
            section = create_loft_profile(profile_points, rootComp, loftFeats)
            loftInput.loftSections.add(section)

        loftFeats.add(loftInput)

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))