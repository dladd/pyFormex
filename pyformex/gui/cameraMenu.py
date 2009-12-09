#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Tue Dec  8 12:25:08 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Camera handling menu"""

import pyformex as GD
import draw
import toolbar
from gettext import gettext as _


         
def zoomIn():
    GD.canvas.camera.zoomArea(1./float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()
def zoomOut():
    GD.canvas.camera.zoomArea(float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()
def panRight():
    GD.canvas.camera.transArea(-float(GD.cfg['gui/panfactor']),0.)
    GD.canvas.update()   
def panLeft():
    GD.canvas.camera.transArea(float(GD.cfg['gui/panfactor']),0.)
    GD.canvas.update()   
def panUp():
    GD.canvas.camera.transArea(0.,-float(GD.cfg['gui/panfactor']))
    GD.canvas.update()   
def panDown():
    GD.canvas.camera.transArea(0.,float(GD.cfg['gui/panfactor']))
    GD.canvas.update()   
def rotRight():
    GD.canvas.camera.rotate(+float(GD.cfg['gui/rotfactor']),0,1,0)
    GD.canvas.update()   
def rotLeft():
    GD.canvas.camera.rotate(-float(GD.cfg['gui/rotfactor']),0,1,0)
    GD.canvas.update()   
def rotUp():
    GD.canvas.camera.rotate(-float(GD.cfg['gui/rotfactor']),1,0,0)
    GD.canvas.update()   
def rotDown():
    GD.canvas.camera.rotate(+float(GD.cfg['gui/rotfactor']),1,0,0)
    GD.canvas.update()   
def twistLeft():
    GD.canvas.camera.rotate(+float(GD.cfg['gui/rotfactor']),0,0,1)
    GD.canvas.update()   
def twistRight():
    GD.canvas.camera.rotate(-float(GD.cfg['gui/rotfactor']),0,0,1)
    GD.canvas.update()   
def transLeft():
    val = float(GD.cfg['gui/panfactor']) * GD.canvas.camera.getDist()
    GD.canvas.camera.translate(-val,0,0,GD.cfg['draw/localaxes'])
    GD.canvas.update()   
def transRight():
    val = float(GD.cfg['gui/panfactor']) * GD.canvas.camera.getDist()
    GD.canvas.camera.translate(+val,0,0,GD.cfg['draw/localaxes'])
    GD.canvas.update()   
def transDown():
    val = float(GD.cfg['gui/panfactor']) * GD.canvas.camera.getDist()
    GD.canvas.camera.translate(0,-val,0,GD.cfg['draw/localaxes'])
    GD.canvas.update()   
def transUp():
    val = float(GD.cfg['gui/panfactor']) * GD.canvas.camera.getDist()
    GD.canvas.camera.translate(0,+val,0,GD.cfg['draw/localaxes'])
    GD.canvas.update()   
def dollyIn():
    GD.canvas.camera.dolly(1./float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()   
def dollyOut():
    GD.canvas.camera.dolly(float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()   
def report():
    print(GD.canvas.camera.report())


MenuData = [
    (_('&LocalAxes'),draw.setLocalAxes),
    (_('&GlobalAxes'),draw.setGlobalAxes),
    (_('&Projection'),toolbar.setProjection),
    (_('&Perspective'),toolbar.setPerspective),
    (_('&Zoom All'),draw.zoomAll), 
    (_('&Zoom In'),zoomIn), 
    (_('&Zoom Out'),zoomOut), 
    (_('&Dolly In'),dollyIn), 
    (_('&Dolly Out'),dollyOut), 
    (_('&Pan Left'),panLeft), 
    (_('&Pan Right'),panRight), 
    (_('&Pan Down'),panDown), 
    (_('&Pan Up'),panUp), 
    (_('&Translate'),[
        (_('Translate &Left'),transLeft), 
        (_('Translate &Right'),transRight), 
        (_('Translate &Down'),transDown),
        (_('Translate &Up'),transUp),
        ]),
    (_('&Rotate'),[
        (_('Rotate &Left'),rotLeft),
        (_('Rotate &Right'),rotRight),
        (_('Rotate &Down'),rotDown), 
        (_('Rotate &Up'),rotUp),
        (_('Rotate &ClockWise'),twistRight),
        (_('Rotate &CCW'),twistLeft),
        ]),
    ('---',None),
    (_('&Report'),report), 
    ]


# End
