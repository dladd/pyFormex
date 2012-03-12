# $Id$
##
##  This file is part of pyFormex
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


"""Icons

Create an icon file from a pyFormex model rendering.

This application was used to create some of the toolbar icons for pyFormex
"""
_status = 'checked'
_level = 'normal'
_topics = ['geometry']
_techniques = ['image','icon']

from gui.draw import *

from apps.Clock import AnalogClock
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
    view('front')
    F = AnalogClock()
    F.draw()
    F.drawTime(11,55)


def icon_run():
    view('front')
    F = Formex('3:016045').trl([-0.3,0.,0.]
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


def icon_script():
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
