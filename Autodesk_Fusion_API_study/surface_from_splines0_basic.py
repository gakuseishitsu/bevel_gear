import adsk.core, adsk.fusion, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface   
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)# Create a document.
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent

        # Create profile 1
        sketch0 = rootComp.sketches.add(rootComp.xZConstructionPlane)
        points = adsk.core.ObjectCollection.create()
        points.add(adsk.core.Point3D.create(0, 0, 0))
        points.add(adsk.core.Point3D.create(5, 1, 0))
        points.add(adsk.core.Point3D.create(6, 4, 3))
        curve0 = sketch0.sketchCurves.sketchFittedSplines.add(points)

        # Create profile 2
        sketch1 = rootComp.sketches.add(rootComp.xZConstructionPlane)
        points = adsk.core.ObjectCollection.create()
        points.add(adsk.core.Point3D.create(0, 0, 1))
        points.add(adsk.core.Point3D.create(5, 1, 1))
        points.add(adsk.core.Point3D.create(6, 4, 4))
        curve1 = sketch1.sketchCurves.sketchFittedSplines.add(points)

        # Create profile 3
        sketch2 = rootComp.sketches.add(rootComp.xZConstructionPlane)
        points = adsk.core.ObjectCollection.create()
        points.add(adsk.core.Point3D.create(0, 0, 2))
        points.add(adsk.core.Point3D.create(5, 1, 2))
        points.add(adsk.core.Point3D.create(6, 4, 5))
        curve2 = sketch2.sketchCurves.sketchFittedSplines.add(points)

        ### Create loft feature input
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        oftSectionsObj = loftInput.loftSections
        loftInput.isSolid = False
        loftInput.isClosed = False
        loftInput.isTangentEdgesMerged = True

        oftSectionsObj.add(rootComp.features.createPath(curve0))
        oftSectionsObj.add(rootComp.features.createPath(curve1))
        oftSectionsObj.add(rootComp.features.createPath(curve2))

        loftFeats.add(loftInput)

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))