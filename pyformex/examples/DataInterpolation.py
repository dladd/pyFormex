#!/usr/bin/pyformex
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

# First, we need to import calpy. If you do not have calpy,
# download it from ftp://bumps.ugent.be/pub/calpy
# and compile/install it

# Locate calpy and load interface
from plugins import calpy_itf


# Now, let's create a grid of 'quad8' elements
# size of the grid
nx,ny = 4,3
# plexitude
nplex = 8
clear()
flatwire()
M = Formex(mpattern('123')).replic2(nx,ny).toMesh().convert('quad%s'%nplex,fuse=True)
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

#End
