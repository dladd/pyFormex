#!/usr/bin/env pyformex
# $Id$
# Creates random points, bars, triangles, quads, ...
"""Random"""
from numarray import random_array
npoints = 100
P = Formex(random_array.random((npoints,1,3)))
clear();drawProp(P);sleep()
for n in range(2,4):
    F = Formex.connect([P for i in range(n)],bias=[i*(n-1) for i in range(n)],loop=True)
    F.setProp(arange(npoints))
    clear();drawProp(F);sleep()
