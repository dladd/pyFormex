#!/usr/bin/env python pyformex.py
# $Id$
#
"""Barrel Vault"""
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

print m
print n
da = d.replic2(m,n,2,2)
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

# Create barrel
grid = (da+ha+ea).rotate(90,1).translate(0,r)
draw(grid)

grid = grid.scale([1.,pi*r/(2*n),l/(2*m)])
draw(grid)

barrel = grid.cylindrical(scale=[1.,a/(pi*r),1.])
draw(barrel)

# That's all, folks!
