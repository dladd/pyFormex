# $Id$

"""ExtrudeBorder

This example illustrates some surface techniques. A closed surface is cut
with a number(3) of planes. Each cut leads to a hole, the border of which is
then extruded over a gioven length in the direction of the plane's positive
normal. 
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['extrude','borderfill','cut']

from gui.draw import *

def cutBorderClose(S,P,N):
    """Cut a surface with a plane, and close it

    Return the border line and the closed surface.
    """
    S = S.cutWithPlane(P,N,side='-')
    B = S.border()[0]
    return B,S.close()
  

def run():
    import simple
    smooth()
    linewidth(2)
    clear()
    S = simple.sphere()
    SA = draw(S)

    p = 0
    for P,N,L,ndiv in [
        #
        # Each line contains a point, a normal, an extrusion length
        # and the number of elements along this length
        ((0.6, 0., 0.), (1., 0., 0.), 2.5, 5 ),
        ((-0.6, 0.6, 0.), (-1., 1., 0.), 4., 16),
        ((-0.6, -0.6, 0.), (-1., -1., 0.), 3., 2),
        ]:
        B,S = cutBorderClose(S,P,N)
        draw(B)
        p += 1
        E = B.extrude(n=ndiv,step=L/ndiv,dir=normalize(N),eltype='tri3').setProp(p)
        draw(E)
        
    draw(S)
    undraw(SA)
    zoomAll()


if __name__ == 'draw':
    run()
# End
