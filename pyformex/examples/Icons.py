# $Id$    *** pyformex ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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


"""Icons

Create an icon file from a pyFormex model rendering.

This application was used to create some of the toolbar icons for pyFormex
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['geometry']
_techniques = ['image','icon']

from gui.draw import *

from gui.actors import CubeActor
from gui.image import saveIcon
from plugins.curve import *


def icon_smooth():
    view('iso')
    F = CubeActor()
    drawActor(F)
    smooth()
    zoom(0.8)
    

def icon_clock():
    from examples.Clock import AnalogClock
    view('front')
    F = AnalogClock()
    F.draw()
    F.drawTime(11,55)


def icon_run():
    view('front')
    F = Formex('3:016045').trl([-0.3,0.,0.])
    draw(F)


def icon_rerun():
    icon_run()
    A = Arc(radius=1.5,angles=(45.,135.)).setProp(1)
    B = A.scale(0.8)
    MA = A.approx().toMesh()
    MB = B.approx().toMesh()
    C = MA.connect(MB)
    draw(C)
    D = F.scale(0.7).rotate(-45).setProp(1).trl(A.coords[0].scale(0.9))
    draw(D)
    E = C.rotate(180)
    F = D.rotate(180)
    draw([E,F])
    zoomAll()


def spiral(X,dir=[0,1,2],rfunc=lambda x:1,zfunc=lambda x:1):
    """Perform a spiral transformation on a coordinate array"""
    theta = X[...,dir[0]]
    r = rfunc(theta) + X[...,dir[1]]
    x = r * cos(theta)
    y = r * sin(theta)
    z = zfunc(theta) + X[...,dir[2]]
    X = hstack([x,y,z]).reshape(X.shape)
    return Coords(X)

def icon_reset():
    T = Formex([[(0,0),(-3,0),(-3,3)]])
    draw(T,color='steelblue')
    x = Coords([(-2,2),(-1,3),(3,3),(3,0)])
    draw(x)
    P = BezierSpline(control=x)
    x = Coords([(3,0),(3,-1),(3,-2),(1,-3)])
    draw(x)
    P1 = BezierSpline(control=x)
    draw([P,P1],color='indianred')
    zoomAll()


def icon_script():
    icon_run()
    from examples import FontForge
    okfonts = [ f for f in FontForge.fonts if 'Sans' in f and 'Oblique' in f ]
    res = askItems([
        _I('fontname',None,choices=okfonts),
        ])
    if res:
        fontname = res['fontname']
        curve = FontForge.charCurve(fontname,'S')
        curve = curve.scale(2.5/curve.sizes()[1]).centered()
        FontForge.drawCurve(curve,color=red,fill=True,with_border=False,with_points=False)
        print(curve.bbox())
    zoomAll()


def available_icons():
    """Create a list of available icons.

    The icon name is the second part of the 'icon_' function names.
    """
    icons = [ i[5:] for i in globals().keys() if i.startswith('icon_') and callable(globals()[i]) ]
    icons.sort()
    return icons


def run():

    resetAll()
    flat()
    bgcolor('white') # Make sure we have a one color background
    

    res = askItems([
        _I('icon',text='Icon Name',choices=_avail_icons),
        _I('save',False,text='Save Icon'),
        ])
    
    if not res:
        return

    icon = res['icon']
    save = res['save']

    create = globals()['icon_'+icon]
    create()


    if save:
        saveIcon(icon)
        

_avail_icons = available_icons()
       

if __name__ == 'draw':
    run()
# End
