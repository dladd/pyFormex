#!/usr/bin/env pyformex
# $Id$

"""Koch line"""

from lima import *
import math
# We use the lima module to create six generations of the Koch line
F = [ Formex(lima("F",{"F":"F*F//F*F"},i,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })) for i in range(6) ]
# and display them in series
clear()
drawProp(Formex.concatenate([F[i].scale(math.pow(3,5-i)).translate([0,i*60,0]) for i in range(6)]))
# a variant which dispays the lines as radii of a six-pointed star
#clear()
#drawProp(Formex.concatenate([F[i].rotate(60*i).scale(math.pow(3,5-i)) for i in range(6)]))
