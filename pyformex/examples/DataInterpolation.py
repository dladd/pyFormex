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
#
"""DataInterpolation

level = 'advanced'
topics = [ 'mesh','postprocess']
techniques = ['calpy','color']

.. Description

DataInterpolation
-----------------

This example demonstrates how to use calpy to interpolate data on a Mesh.
The data are given in the Gauss integration points (GP) and are
computed in the nodes, but the same technique can be used if data
are given in other points or need to be calculated in other points.
"""
_status = 'unchecked'
_level = 'advanced'
_topics = [ 'mesh','postprocess']
_techniques = ['calpy','color']

from gui.draw import *

# First, we need to import calpy. If you do not have calpy,
# download it from ftp://bumps.ugent.be/pub/calpy
# and compile/install it

# Locate calpy and load interface
from plugins import calpy_itf

def run():
    # Now, let's create a grid of 'quad8' elements
    # size of the grid
    nx,ny = 4,3
    # plexitude
    nplex = 8
    clear()
    flatwire()
    M = Formex('4:0123').replic2(nx,ny).toMesh().convert('quad%s'%nplex,fuse=True)
    #draw(M,color=yellow)

    # Create the Mesh interpolateor
    gprule = (5,1) # integration rule: minimum (1,1),  maximum (5,5)
    Q = calpy_itf.QuadInterpolator(M.nelems(),M.nplex(),gprule)

    # Define some random data at the GP.
    # We use 3 data per GP, because we will use the data directly as colors
    ngp = prod(gprule) # number of datapoints per element
    data = random.rand(M.nelems(),ngp,3)
    print "Number of data points per element: %s" % ngp
    print "Original element data: %s" % str(data.shape)
    # compute the data at the nodes, per element
    endata = Q.GP2Nodes(data)
    print "Element nodal data: %s" % str(endata.shape)
    # compute nodal averages
    nodata = Q.NodalAvg(M.elems+1,endata,M.nnodes())
    print "Average nodal data: %s" % str(nodata.shape)
    # extract the colors per element
    colors = nodata[M.elems]
    print "Color data: %s" % str(colors.shape)
    layout(2)

    viewport(0)
    clear()
    flatwire()
    draw(M,color=endata)
    drawNumbers(M.coords)
    drawText("Per element interpolation",20,20,font='9x15')

    viewport(1)
    clear()
    flatwire()
    draw(M,color=colors)
    drawNumbers(M.coords)
    drawText("Averaged nodal values",20,20,font='9x15')

if __name__ == 'draw':
    run()
# End
