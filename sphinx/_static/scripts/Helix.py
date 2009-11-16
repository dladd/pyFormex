#!/usr/bin/env pyformex
"""Helix example from pyFormex"""
m = 36 # number of cells along helix
n = 10 # number of cells along circular cross section
reset()
setDrawOptions({'clear':True})
F = Formex(pattern("164"),[1,2,3]); draw(F)
F = F.replic(m,1.,0); draw(F)
F = F.replic(n,1.,1); draw(F)
F = F.translate(2,1.); draw(F,view='iso')
F = F.cylindrical([2,1,0],[1.,360./n,1.]); draw(F)
F = F.replic(5,m*1.,2); draw(F)
F = F.rotate(-10.,0); draw(F)
F = F.translate(0,5.); draw(F)
F = F.cylindrical([0,2,1],[1.,360./m,1.]); draw(F)
draw(F,view='right')
