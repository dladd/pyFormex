#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
#
"""Torus"""
def torus(m,n):
    """Create a torus with m cells along big circle and n cells along small."""
    F = Formex(pattern("164"),[1,2,3]).replic2(m,n,1,1)
    G = F.translate1(2,1).cylindrical([2,1,0],[1.,360./n,1.])
    H = G.translate1(0,5).cylindrical([0,2,1],[1.,360./m,1.])
    return H

for m in [3,4,6,12,36]:
    side='front'
    for n in [3,4,6,8,12]:
        clear()
        draw(torus(m,n),side)
        side=None
        sleep()

clear()
draw(torus(72,36),None)
