#!/usr/bin/env python pyformex.py
# $Id$
#
"""Torus variants

level = 'normal'
topics = ['geometry']
techniques = ['programming','widgets','globals']

"""

def torus(m,n,surface=True):
    """Create a torus with m cells along big circle and n cells along small."""
    if surface:
        C = Formex([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],[1,3])
    else:
        C = Formex(pattern("164"),[1,2,3])
    F = C.replic2(m,n,1,1)
    G = F.translate(2,1).cylindrical([2,1,0],[1.,360./n,1.])
    H = G.translate(0,5).cylindrical([0,2,1],[1.,360./m,1.])
    return H


def series():
    view='iso'
    for n in [3,4,6,8,12]:
        for m in [3,4,6,12,36]:
            clear()
            draw(torus(m,n),view)
            view=None
        
def drawTorus(m,n):
    clear()
    draw(torus(m,n),None)
    
def nice():
    drawTorus(72,36)


m = 20
n = 10
while True:
    res = askItems([('m',m,'slider',{'min':3,'max':72}),
                    ('n',n,'slider',{'min':3,'max':36})
                    ])
    if not res:
        break
    globals().update(res)
    drawTorus(m,n)

# End
