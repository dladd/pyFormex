#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
"""Spiral"""
# This constructs the same example as torus.py, but shows all steps
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle
# First create a long rectangle
def drawit(F,side='front'):
    clear()
    drawProp(F,side)
    sleep()
F = Formex(pattern("164"),[1,2,3]); drawit(F)
F = F.replicate(m,0,1); drawit(F)
F = F.replicate(n,1,1); drawit(F)
F = F.translate1(2,1); drawit(F,'iso')
F = F.cylindrical([2,1,0],[1.,360./n,1.]); drawit(F,'iso')
F = F.replicate(5,2,m); drawit(F,'iso')
F = F.rotate(-10,0); drawit(F,'iso')
F = F.translate1(0,5); drawit(F,'iso')
F = F.cylindrical([0,2,1],[1.,360./m,1.]); drawit(F,'iso')
drawit(F,'right')
