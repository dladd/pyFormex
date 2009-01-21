#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Camera handling menu"""

import pyformex as GD
import draw
import toolbar
from gettext import gettext as _


         
def zoomIn():
    GD.canvas.zoom(1./float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()
def zoomOut():
    GD.canvas.zoom(float(GD.cfg['gui/zoomfactor']))
    GD.canvas.update()
##def panRight():
##    canvas.camera.pan(+5)
##    canvas.update()   
##def panLeft():
##    canvas.camera.pan(-5)
##    canvas.update()   
##def panUp():
##    canvas.camera.pan(+5,0)
##    canvas.update()   
##def panDown():
##    canvas.camera.pan(-5,0)
##    canvas.update()   
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
    (_('&Translate'),[
        (_('Translate &Right'),transRight), 
        (_('Translate &Left'),transLeft), 
        (_('Translate &Up'),transUp),
        (_('Translate &Down'),transDown),
        ]),
    (_('&Rotate'),[
        (_('Rotate &Right'),rotRight),
        (_('Rotate &Left'),rotLeft),
        (_('Rotate &Up'),rotUp),
        (_('Rotate &Down'),rotDown), 
        (_('Rotate &ClockWise'),twistRight),
        (_('Rotate &CCW'),twistLeft),
        ]),
    ]


# End
