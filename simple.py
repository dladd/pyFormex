#!/usr/bin/env python
# $Id$
#
"""Predefined Formex samples with simple geometric shapes."""

from formex import *

Pattern = {
    'line'   : '1',
    'angle'  : '102',
    'square' : '1234',
    'plus'   : '1020304',
    'cross'  : '5060708',
    'diamond' : '5678',
    'rtriangle' : '164',
    'cube'   : '1234i/Ai/Bi/Ci/41234',
    'star'   : '102030405060708',
    'star3d' : '1/02/03/04/05/06/07/08/0A/0B/0C/0D/0E/0F/0G/0H/0a/0b/0c/0d/0e/0f/0g/0h'
    }


def unitcircle(a1=1.,a2=2.,a3=360.):
    """Return a Formex which is a unit circle at the origin in the x-y-plane.

    a1: dash angle in degrees, a2: modular angle in degrees, a3: total angle.
    a1 == a2 gives a full circle, a1 < a2 gives a dashed circle.
    If a3 < 360, the result is an arc.
    The default values give a dashed circle.
    Large angle values result in polygones.
    """
    n = int(round(a3/a2))
    a1 *= pi/180.
    return Formex([[[1.,0.,0.],[cos(a1),sin(a1),0.]]]).rosette(n,a2,axis=2,point=[0.,0.,0.])
