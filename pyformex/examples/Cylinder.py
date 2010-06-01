#!/usr/bin/env pyformex
# $Id$

"""Cylinder

level = 'beginner'
topics = ['geometry', 'surface']
techniques = ['import']

.. Description

Cylinder
--------
This example illustrates the use of simple.sector() and simple.cylinder()
to create a parametric cylindrical surface.

"""
import simple
from plugins.surface import TriSurface

n=12
h=5.
A = simple.sector(1.,360.,1,n,diag='up')
B = simple.cylinder(2.,h,n,4,diag='u').reverse()
C = A.reverse()+B+A.trl(2,h)
S = TriSurface(C)
export({'surface':S})

smoothwire()
view('iso')
draw(S,color=red)

# End
