#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
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
