# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
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
        _I('far',100,text='Far clipping plane',itemtype='fslider',func=setFar,tooltip='Distance of the far clipping plane to the camera.'),
        ],actions = [('Close',close),
#                     ('Apply',apply),
                     ],
                    default='Close',
                    )

    dialog.show()

    cam.modelview_callback = updateSettings
    

if __name__ == 'draw':
    showCameraTool()
    dialog.timeout = close
    
# End
