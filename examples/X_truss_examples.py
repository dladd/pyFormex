#!/usr/bin/env pyformex
# $Id$
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
