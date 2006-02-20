#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""X-shaped truss"""

from examples.X_truss import X_truss

def example(diag=True,vert=True):
    truss = X_truss(12,2.35,2.65,diag,vert)
            
    truss.bot.setProp(3)
    truss.top.setProp(3)
    truss.vert.setProp(0)
    truss.dia1.setProp(1)
    truss.dia2.setProp(1)

    clear()
    draw(truss.allNodes(),wait=False)
    draw(truss.allBars())

for diag in [True,False]:
    for vert in [True,False]:
        example(diag,vert)


# End
