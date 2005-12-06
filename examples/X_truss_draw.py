#!/usr/bin/env pyformex
# $Id$
#
"""X-shaped truss"""

from examples.X_truss import X_truss

def example(diag=True,vert=True):
    truss = X_truss(12,2.35,2.65,diag,vert)
            
    truss.bot.setProp(0)
    truss.top.setProp(3)
    truss.vert.setProp(2)
    truss.mid1.setProp(1)
    truss.mid2.setProp(5)

    clear()
    draw(truss.allNodes())
    draw(truss.allBars())


for diag in [True,False]:
    for vert in [True,False]:
        example(diag,vert)

