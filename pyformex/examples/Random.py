#!/usr/bin/env pyformex --gui
# $Id$
"""Random

Creates random points, bars, triangles, quads, ...
"""
setDrawOptions(dict(clear=True))
npoints = 30
p = arange(120)
P = Formex(random.random((npoints,1,3)),p)
draw(P,alpha=0.5)
for n in range(2,5):
    F = connect([P for i in range(n)],bias=[i*(n-1) for i in range(n)],loop=True)
    F.setProp(p)
    draw(F,alpha=0.5)

# End
