# $Id$
#

"""Camera tools

Some extra tools to handle the camera.
"""

import pyformex as pf
from gui.widgets import simpleInputItem as _I,InputDialog as Dialog
from utils import stuur

dialog = None

def getCameraSettings(cam):
    return dict([ (k,getattr(cam,k)) for k in ['ctr','dist','rot','fovy','aspect','area','near','far']])
    

def apply():
    global dialog
    dialog.acceptData()
    settings = dialog.results
    #print settings
    cam = pf.canvas.camera
    cam.setClip(settings['near'],settings['far'])
    pf.canvas.update()
    

def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None


def timeOut():
    close()


def updateSettings(cam):
    global dialog
    settings = getCameraSettings(cam)
    dialog.updateData(settings)

def setNear(fld):
    val = fld.value()/100.
    cam = pf.canvas.camera
    res = stuur(val,[0.,0.5,1.0],[0.01*cam.dist,cam.dist,100.*cam.dist])
    #print "%s = %s" % (val,res)
    cam.setClip(res,cam.far)
    pf.canvas.update()
def setFar(fld):
    val = fld.value()/100.
    cam = pf.canvas.camera
    res = stuur(val,[0.,0.5,1.0],[0.01*cam.dist,cam.dist,100.*cam.dist])
    #print "%s = %s" % (val,res)
    cam.setClip(cam.near,res)
    pf.canvas.update()


def showCameraTool():
    global dialog
    cam = pf.canvas.camera
    settings = getCameraSettings(cam)
    settings['near'] = cam.near/cam.dist
    settings['far'] = cam.far/cam.dist

    dialog = Dialog(store=settings, items=[
        _I('ctr',text='Center',itemtype='point',tooltip='The center of the scene where the camera is looking at.'),
        _I('dist',text='Distance',tooltip='The distance of the camera to the center of the scene.'),
        _I('fovy',text='Field of View',tooltip='The vertical opening angle of the camera lens.'),
        _I('aspect',text='Aspect ratio',tooltip='The ratio of the vertical over the horizontal lens opening angles.'),
#        _I('area',text='Visible area',tooltip='Relative part of the camera area that is visible in the viewport.'),
        _I('near',text='Near clipping plane',itemtype='fslider',func=setNear,tooltip='Distance of the near clipping plane to the camera.'),
        _I('far',text='Far clipping plane',itemtype='fslider',func=setFar,tooltip='Distance of the far clipping plane to the camera.'),
        ],actions = [('Close',close),
#                     ('Apply',apply),
                     ],
                    default='Close',
                    )

    dialog.timeout = timeOut
    dialog.show()

    cam.modelview_callback = updateSettings
    

if __name__ == 'draw':
    showCameraTool()
    
# End
