# $Id$  *** pyformex ***
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

"""NurbsDecompose

Illustrates some special techniques on Nurbs Curves:

- inserting knots
- curve decomposing
"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['Geometry', 'Curve']
_techniques = ['nurbs']

from gui.draw import *
from plugins.nurbs import *
from plugins.curve import *
from plugins.nurbs_menu import _options, drawNurbs

class _decors:
    ctrl_numbers = None


def clearDecors():
    undraw(_decors.ctrl_numbers)


def drawNurbs(N):
    clearDecors()
    draw(N,color=_options.color,nolight=True)
    if _options.ctrl:
        draw(N.coords.toCoords(),color=_options.color,nolight=True)
        if _options.ctrl_polygon:
            draw(PolyLine(N.coords.toCoords()),color=_options.color,nolight=True)
        if _options.ctrl_numbers:
            _decors.ctrl_numbers = drawNumbers(N.coords.toCoords())
    if _options.knots:
        draw(N.knotPoints(),color=_options.color,marksize=_options.knotsize)
        if _options.knot_numbers:
            drawNumbers(N.knotPoints())
        if _options.knot_values:
            drawMarks(N.knotPoints(),["%f"%i for i in N.knots],leader='  --> ')


def run():
    clear()
    flat()

    C = Formex('12141214').toCurve()
    #C = Formex('214').toCurve()
    degree = 4

    clear()
    linewidth(1)
    _options.ctrl = True
    _options.ctrl_numbers = True
    _options.ctrl_polygon = True
    _options.knot_values = True

    N = NurbsCurve(C.coords,degree=degree)#,blended=False)
    print(N)
    _options.linewidth = 1
    _options.color = magenta
    _options.knotsize = 5
    drawNurbs(N)
    zoomAll()

    while True:
        res = askItems([
            _I('u',0.2,text='New knot values'),
            ], default='Cancel')
        if not res:
            break;

        u = eval('[%s]' % res['u'])
        N = N.insertKnots(u)
        _options.linewidth = 5
        _options.color = blue
        _options.knotsize = 10
        drawNurbs(N)
        zoomAll()

    ## if ack("Remove knots?"):
    ##     u = 0.5
    ##     print N.removeKnots(u,1,0.001)

        # Break from endless loop if an input timeout is active !
        if widgets.input_timeout >= 0:
            break


    if ack("Decompose curve?"):

        N1 = N.decompose()
        print(N1)
        _options.linewidth = 5
        _options.color = red
        _options.knotsize = 10
        _options.knot_values = False
        drawNurbs(N1)
        zoomAll()
        C = BezierSpline(control=N1.coords.toCoords(),degree=N1.degree)
        draw(C,color=blue)


        if ack("Shift decomposed parts?"):
            CS = [ c.trl([0.,0.1*(-1)**i,0.]) for i,c in enumerate(C.split()) ]
            draw(CS)


if __name__ == 'draw':
    run()

# End
