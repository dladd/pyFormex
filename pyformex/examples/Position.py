#!/usr/bin/env pyformex --gui
# $Id$

"""Position

level = 'beginner'
topics = ['geometry']
techniques = ['position']

Position an object A thus that its three points X are aligned with the
three points X of object B.
"""

out = grepSource('showText')
showText(out,mono=True)
exit()


def drawObjectWithName(obj,name):
    """Draw an object and show its name at the center"""
    drawText3D(obj.center(),name)
    draw(obj)

def drawPointsNumbered(pts,color,prefix):
    """Draw a set of points with their number"""
    draw(pts,color=color,ontop=True,nolight=True)
    drawNumbers(Coords(pts),leader=prefix)


clear()
smoothwire()

# The object to reposition
A = Formex('4:0123',1).replic2(6,3)
# The object to define the position
B = Formex('3:016',2).replic2(4,4,taper=-1).trl(0,7.)

drawObjectWithName(A,'Object A')
drawObjectWithName(B,'Object B')

#define matching points

X = A[0,[0,3,1]]
drawPointsNumbered(X,red,'X')

Y = B[3,[1,2,0]]
Y[2] = Y[0].trl([0.,1.,1.])
drawPointsNumbered(Y,green,'Y')
zoomAll()

pause()

# Reposition A so that X are aligned with Y
C = A.position(X,Y)
draw(C,color=blue)
zoomAll()

# End
