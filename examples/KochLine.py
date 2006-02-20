#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##

"""Koch line"""

from lima import *
import math
# We use the lima module to create six generations of the Koch line
F = [ Formex(lima("F",{"F":"F*F//F*F"},i,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })) for i in range(6) ]
# and display them in series
clear()
draw(Formex.concatenate([F[i].scale(math.pow(3,5-i)).translate([0,i*60,0]) for i in range(6)]))
# a variant which dispays the lines as radii of a six-pointed star
#clear()
#draw(Formex.concatenate([F[i].rotate(60*i).scale(math.pow(3,5-i)) for i in range(6)]))
