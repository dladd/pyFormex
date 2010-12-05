#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
#
"""Barrel Vault

level = 'beginner'
topics = ['geometry']
techniques = ['stepmode','cylindrical'] 
"""

clear()

m=10 # number of modules in axial direction
n=8 # number of modules in tangential direction
r=10. # barrel radius
a=180. # barrel opening angle
l=30. # barrel length


# Diagonals
d = Formex([[[0.,0.,0.],[1.,1.,0.]]],1) # a single diagonal
draw(d,view='front')

d += d.reflect(0,1.) # reflect in x-direction and add to the original
draw(d)

d += d.reflect(1,1.) # reflect in y-direction
draw(d)

da = d.replic(m,2,0) # replicate in x-direction
draw(da)

da = da.replic(n,2,1) # replicate in y-direction
draw(da)

# Longitudinals
h = Formex(pattern("1"),3) # same as  Formex([[[0.,0.,0.],[1.,0.,0.]]],3)
draw(h)

ha = h.replic2(2*m,2*n+1,1,1) # replicate in x- and y-direction
draw(ha)

# End bars
e = Formex(pattern("2"),0) # a unit vertical line
draw(e)

ea = e.replic2(2,2*n,2*m,1) # verticals only at the ends!
draw(ea)

# Choose better viewing angle for 3D
view('iso')
drawAxes()

# Rotate the grid to (y,z) plane and give it an offset from the z-axis
grid = (da+ha+ea).rotate(90,1).translate(0,r)
draw(grid)

# Scale the grid to the requested length and circumference of the barrel
# The current height of the grid is 2*n
# As the angle a is given in degrees, the circumference is
circum = a*Deg*r
scaled_grid = grid.scale([1.,circum/(2*n),l/(2*m)])
draw(scaled_grid)

# Create barrel
# The cylindrical transformation by default expects angles in degrees
barrel = scaled_grid.cylindrical(scale=[1.,(1./r)/Deg,1.])
draw(barrel)
print("Het aantal elementen is %s (plexitude %s)" % (barrel.nelems(),barrel.nplex()))
print("De grootte van de coordinatenarray is %s" % str(barrel.shape()))

# Remark: if we did not want to show the scaled grid, the creation
# of the barrel could be simplified by combining the last two transformations:
# barrel = grid.cylindrical(scale=[1.,a/(2*n),l/(2*m)])


# That's all, folks!
