# $Id$ *** pyformex ***
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

"""NurbsSurface

level = 'advanced'
topics = ['geometry', 'surface']
techniques = ['nurbs']

.. Description

Nurbs
=====
This example is under development.
"""
from gui.draw import *
from plugins.nurbs import *

def run():
    clear()
    smooth()

    createView('myview',angles=(0,-15,0),addtogui=True)
    view('myview')

    #############################
    ####   DATA
    #############################

    # size of the control point grid
    nx,ny = 6,4

    # degree of the NURBS surface
    px,py = 3,2

    # To create a 3D surface, we add z-elevation to some points of the grid
    # The list contains tuples of grid position (x,y) and z-value of peaks
    peaks = [
        (1, 1, 3.),
        (2, 2, -2.)
        ]

    # number of isoparametric curves (-1) to draw on the surface
    kx,ky = 10,4  

    # number of random points
    nP = 100

    # what to draw
    res = askItems([
        ('draw_points',True),
        ('draw_surf',True),
        ('draw_curves',False),
        ('draw_curvepoints',False),
        ('draw_isocurves',True),
        ('draw_randompoints',True),
        ])
    if not res:
        exit()

    globals().update(res)

    ###########################
    ####   CONTROL GRID
    ###########################

    # create the grid of control points
    X = Formex(origin()).replic2(nx,ny).coords.reshape(ny,nx,3)
    for x,y,v in peaks:
        X[x,y,2] = v

    if draw_points:
        # draw the numbered control points
        draw(X,nolight=True)
        drawNumbers(X.reshape(-1,3),trl=[0.05,0.05,0.0])

    ###########################
    ####   NURBS SURFACE
    ###########################

    # create the Nurbs surface
    S = NurbsSurface(X,degree=(px,py))
    if draw_surf:
        # draw the Nurbs surface, with random colors
        colors = 0.5*random.rand(*S.coords.shape)
        draw(S,color=colors[...,:3])

    #exit()

    ###########################
    ####   ISOPARAMETRIC CURVES
    ###########################

    # define isoparametric values for the isocurves
    u = uniformParamValues(kx) # creates kx+1 u-values
    v = uniformParamValues(ky) 

    # create Nurbs curves through 1-d sets of control points, in both directions
    Cu = [NurbsCurve(X[i],degree=px,knots=S.uknots) for i in range(ny)]
    Cv = [NurbsCurve(X[:,i],degree=py,knots=S.vknots) for i in range(nx)]
    if draw_curves:
        # draw the Nurbs curves
        draw(Cu,color=red,nolight=True,ontop=True)
        draw(Cv,color=blue,nolight=True,ontop=True)

    # get points on the Nurbs curves at isoparametric values
    CuP = Coords.concatenate([ Ci.pointsAt(u) for Ci in Cu ]).reshape(ny,kx+1,3)
    CvP = Coords.concatenate([ Ci.pointsAt(v) for Ci in Cv ]).reshape(nx,ky+1,3)
    if draw_curvepoints:
        # draw the isoparametric points
        draw(CuP,marksize=10,ontop=True,nolight=True,color=red)
        drawNumbers(CuP,color=red)
        draw(CvP,marksize=10,ontop=True,nolight=True,color=blue)
        drawNumbers(CvP,color=blue)

    # Create the isocurves: they are Nurbs curves using the isoparametric points
    # in the cross direction as control points
    # First swap the isoparametric point grids, then create curves
    PuC = CuP.swapaxes(0,1)
    PvC = CvP.swapaxes(0,1)
    Vc = [NurbsCurve(PuC[i],degree=py,knots=S.vknots) for i in range(kx+1)] 
    Uc = [NurbsCurve(PvC[i],degree=px,knots=S.uknots) for i in range(ky+1)]
    if draw_isocurves:
        # draw the isocurves
        draw(Vc,color=red,linewidth=2,nolight=True)#,ontop=True)
        color = 0.5*random.rand(Uc[0].coords.shape[0],3)
        print color.shape
        print color
        draw(Uc,color=[red,yellow,green,cyan,blue,magenta],linewidth=3,nolight=True)#,ontop=True)



    ###########################
    ####   RANDOM POINTS
    ###########################

    # create random parametric values and compute points on the surface
    u = random.random(2*nP).reshape(-1,2)
    P = S.pointsAt(u)
    if draw_randompoints:
        # draw the random points
        draw(P,color=black,nolight=True,ontop=True)


if __name__ == 'draw':
    run()
# End
