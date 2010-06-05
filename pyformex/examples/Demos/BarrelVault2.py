#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

level = 'advanced'
topics = ['FEA']
techniques = ['colors'] 
"""

clear()
m=10 # number of modules in axial direction
n=8 # number of modules in tangential direction
r=10. # barrel radius
a=180. # barrel opening angle
l=30. # barrel length


# Diagonals
d = Formex([[[0.,0.,0.],[1.,1.,0.]]],1)
draw(d,view='front')
d += d.reflect(0,1) # reflect in x-direction
d += d.reflect(1,1) # reflect in y-direction
draw(d)

# Replicate in x-direction
da = d.replic(m,2,0)
draw(da)
# Replicate in y-direction
da = da.replic(n,2,1)
draw(da)

# Longitudinals
h = Formex(pattern("1"),3) # Same as  Formex([[[0.,0.,0.],[1.,0.,0.]]],3)
draw(h)
ha = h.replic2(2*m,2*n+1,1,1)
draw(ha)

# End bars
e = Formex(pattern("2"),0)
draw(e)
ea = e.replic2(2,2*n,2*m,1)
draw(ea)


view('iso')
# Create barrel
grid = (da+ha+ea).rotate(90,1).translate(0,r)
draw(grid)

grid = grid.scale([1.,pi*r/(2*n),l/(2*m)])
draw(grid)

barrel = grid.cylindrical(scale=[1.,a/(pi*r),1.])
draw(barrel)

# That's all, folks!
