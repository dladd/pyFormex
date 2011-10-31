#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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
"""Spirals

level = 'normal'
topics = ['geometry','curve','mesh']
techniques = ['sweep',]
"""
from plugins import curve
from odict import ODict

import simple
import re

linewidth(2)
clear()

rfuncs = [
    'linear (Archimedes)',
    'quadratic',
    'exponential (equi-angular)',
    'constant',
#    'custom',
]

# Define a dictionary of planar cross sections
cross_sections_2d = ODict()
# select the planar patterns from the simple module
for cs in simple.Pattern:
    if re.search('[a-zA-Z]',simple.Pattern[cs]) is None:
        cross_sections_2d[cs] = simple.Pattern[cs]
# add some more patterns
cross_sections_2d.update({
    'swastika':'l:12+23+34+41',
    'channel' : 'l:1223',
    'H-beam' : 'l:11/322/311',
    'sigma' : 'l:16253',
    'octagon':'l:15263748',
    'Z-beam': 'l:353',
    'solid_square': '4:0123',
    'solid_triangle': '3:012',
    'swastika3': '3:012023034041',
    })


input_data = [
    _I('nmod',100,text='Number of cells along spiral'),
    _I('turns',2.5,text='Number of 360 degree turns'),
    _I('rfunc',None,text='Spiral function',choices=rfuncs),
    _I('coeffs',(1.,0.5,0.2),text='Coefficients in the spiral function'),
    _I('spiral3d',0.0,text='Out of plane factor'),
    _I('spread',False,text='Spread points evenly along spiral'),
    _I('nwires',1,text='Number of spirals'),
    _G('sweep',text='Sweep Data',checked=True,items= [
        _I('cross_section','cross','select',text='Shape of cross section',choices=cross_sections_2d.keys()),
        _I('cross_rotate',0.,text='Cross section rotation angle before sweeping'),
        _I('cross_upvector','2',text='Cross section vector that keeps its orientation'),
        _I('cross_scale',0.,text='Cross section scaling factor'),
        ]),
    _I('flyalong',False,text='Fly along the spiral'),
   ]


def spiral(X,dir=[0,1,2],rfunc=lambda x:1,zfunc=lambda x:0):
    """Perform a spiral transformation on a coordinate array"""
    theta = X[...,dir[0]]
    r = rfunc(theta) + X[...,dir[1]]
    x = r * cos(theta)
    y = r * sin(theta)
    z = zfunc(theta) + X[...,dir[2]]
    X = hstack([x,y,z]).reshape(X.shape)
    return Coords(X)


def drawSpiralCurves(PL,nwires,color1,color2=None):
    if color2 is None:
        color2 = color1
    # Convert to Formex, because that has a rosette() method
    PL = PL.toFormex()
    if nwires > 1:
        PL = PL.rosette(nwires,360./nwires)
    draw(PL,color=color1)
    draw(PL.points(),color=color2)


def createCrossSection():
    CS = Formex(cross_sections_2d[cross_section])
    if cross_rotate :
        CS = CS.rotate(cross_rotate)
    if cross_scale:
        CS = CS.scale(cross_scale)
    # Convert to Mesh, because that has a sweep() method
    CS = CS.swapAxes(0,2).toMesh()
    return CS


def createSpiralCurve(turns,nmod):
    F = Formex(origin()).replic(nmod,1.,0).scale(turns*2*pi/nmod)
    a,b,c = coeffs
    rfunc_defs = {
        'constant':                    lambda x: a,
        'linear (Archimedes)':         lambda x: a + b*x,
        'quadratic' :                  lambda x: a + b*x + c*x*x,
        'exponential (equi-angular)' : lambda x: a + b * exp(c*x),
#        'custom' :                     lambda x: a + b * sqrt(c*x),
    }

    rf = rfunc_defs[rfunc]
    if spiral3d:
        zf = lambda x : spiral3d * rf(x)
    else:
        zf = lambda x : 0.0

    S = spiral(F.coords,[0,1,2],rf,zf)

    PL = curve.PolyLine(S[:,0,:])

    return PL
    


def show():
    """Accept the data and draw according to them"""
    clear()
    dialog.acceptData()
    globals().update(dialog.results)

    PL = createSpiralCurve(turns,nmod)
    drawSpiralCurves(PL,nwires,red,blue)

    if spread:
        at = PL.atLength(PL.nparts)
        X = PL.pointsAt(at)
        PL = curve.PolyLine(X)
        clear()
        drawSpiralCurves(PL,nwires,blue,red)


    if sweep:

        CS = createCrossSection()
        draw(CS)

        draw(CS)
        wait()
        structure = CS.sweep(PL,normal=[1.,0.,0.],upvector=eval(cross_upvector),avgdir=True)
        clear()
        smoothwire()
        draw(structure,color='red',bkcolor='cyan')

        if nwires > 1:
            structure = structure.toFormex().rosette(nwires,360./nwires).toMesh()
            draw(structure,color='orange')

    if flyalong:
        flyAlong(PL.scale(1.1).trl([0.0,0.0,0.2]),upvector=[0.,0.,1.],sleeptime=0.1)

        view('front')



def close():
    global dialog
    pf.PF['Sweep_data'] = dialog.results
    if dialog:
        dialog.close()
        dialog = None
    scriptRelease(__file__)


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout. This is important for developers.
    Most normal users can simply ignore it.
    """
    show()
    close()


if __name__ == 'draw':

    # Update the data items from saved values
    try:
        saved_data = pf.PF.get('Sweep_data',{})
        #
        widgets.updateDialogItems(input_data,saved_data)
    except:
        raise


    # Create the modeless dialog widget
    dialog = widgets.InputDialog(input_data,caption='Sweep Dialog',actions = [('Close',close),('Show',show)],default='Show')

    # Examples style requires a timeout action
    dialog.timeout = timeOut

    # Show the dialog and let the user have fun
    dialog.show()
    scriptLock(__file__)

# End

