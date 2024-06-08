import adsk.core, adsk.fusion, adsk.cam, traceback
import os, sys

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        install_numpy = sys.path[0] +'\Python\python.exe -m pip install numpy'
        install_scipy = sys.path[0] +'\Python\python.exe -m pip install scipy'

        os.system('cmd /c "' + install_numpy + '"')
        os.system('cmd /c "' + install_scipy + '"')
        
        try:
            import scipy
            ui.messageBox("Installation succeeded !")
        except:
            ui.messageBox("Failed when executing 'import scipy'")

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))