#!/usr/bin/env pyformex --gui
# $Id$
"""Multicut

level = 'beginner'
topics = ['surface']
techniques = ['cut']

.. Description

Multicut
--------
This example shows how to cut a hole in a surface.
It uses the cutWithPlane function with a series of cutting planes.

"""
clear()

from plugins.trisurface import Sphere
S = Sphere(4).scale(3.)
T = S.cutWithPlane([[2.,0.,0.],[0.,1.,0.],[-2.,0.,0.],[0.,-1.,0.]],
                   [[-1.,0.,0.],[0.,-1.,0.],[1.,0.,0.],[0.,+1.,0.]],
                   side = '-')
draw(T)

# End
