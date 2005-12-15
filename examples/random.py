#!/usr/bin/env pyformex
# $Id$
# Creates random points, bars, triangles, quads, ...
"""Random"""
from scipy import random
npoints = 100
P = Formex(random.random((npoints,1,3)))
clear();draw(P);
for n in range(2,4):
    F = Formex.connect([P for i in range(n)],bias=[i*(n-1) for i in range(n)],loop=True)
    F.setProp(arange(npoints))
    clear();draw(F);
