#!/usr/bin/env pyformex --gui
# $Id$
#

"""Rubik

level = 'normal'
topics = ['illustration']
techniques = ['colors','dialog','draw','persistence','random']

.. Description

Rubik
-------

This example illustrates how the user can interact with the canvas.
Herefore, a Rubik cube with a variable number of rows is shown.
By keeping the *SHIFT* key pressed and dragging an element of the cube, one can rotate a row of the cube.
Pressing the *SHIFT* key is needed so that the script does not interfere with basic canvas functionality
like rotating and zooming.

The script creates a nxnxn cube as a Formex. When the left mouse button is used together with the *SHIFT* key,
the definition **turn** is executed. When the LMB is pressed, the current location of the mouse cursor is registered
and the cube element (i.e. a quadrilateral of the Formex) closest by this position is registered. When the LMB is released,
the current position of the cursor is compared to the former position. If the cursor position has changed, the selected
element is rotated in the direction determined by the projection of the vector created by the position change, rotated
over the current view rotation and projected onto the plane of the selected element.

The cubes can be shuffled into a random position and solved subsequently.

The maximum number of cubes in one row is limited to ten because of the enormous number of permutations possible,
and thus the large amount of time needed to solve such large cubes. For a 7x7 cube for example, the total number
of permutations is already higher than the assumed total number of atoms in our universe (eh ... that's something
like ten to the power 80, but we might be way off). You can check the exact number of possible permutations
for the displayed cube by pressing the button *permutations*. This number is not stored, it is calculated each time. It's
an amazing example of how python can easily handle huge numbers.


"""

from formex import *
from gui.widgets import simpleInputItem as I
from gui.viewport import *
from numpy.random import rand
import time
import math


# General definitions
def createCube(n=3):
    tol = 0.005
    front = Formex(mpattern('123')).replic2(n,n).translate([-n/2.,-n/2.,-n/2.])
    sides = front+front.translate(2,n)
    darkPosTol = Formex(mpattern('123')).replic2(n,n).translate([-n/2.,-n/2.,0.]).scale(1.-5.*tol/n).translate(2, -n/2.+1+tol)
    darkNegTol = darkPosTol.translate(2, -2*tol)
    dark = darkPosTol.replic(n-1, 1, dir=2) + darkNegTol.replic(n-1, 1, dir=2)
    cube = sides + sides.rotate(90,1) + sides.rotate(90,0) + dark + dark.rotate(90, 0) + dark.rotate(90, 1)
    cube.prop = append(repeat(array([5,1,3,2,6,7]),n**2), repeat(array([0]), 6*(n-1)*n**2))
    return cube

def refresh():
    """Refresh the cube on the canvas"""
    global drawn
    clear()
    drawn = draw(cube)

# Rotation definitions
def turn(x=0,y=0,action=0):
    """Execute a rotation when SHIFT and LMB is pressed"""
    global busy, x1, y1, element
    if action==PRESS and not busy:
        busy = True
        pf.canvas.setCursorShape('pick')
        x1, y1 = x, y
        busy = False
        element = selectElement(pf.canvas, x, y, 2, 2)
        if element == [-1]:
            message('No element selected.\nPlease select an element of the cube.')
#        else:
#            draw(cube[element], color=red, bbox='last', linewidth=5.0)
        busy = False
    if action==RELEASE and not busy:
        busy = True
        pf.canvas.setCursorShape('default')
        if element != [-1] and x1!=-1:
            x2, y2 = x, y
            dx = float(x2-x1)
            dy = float(y2-y1)
            if dx == 0 and dy == 0:
                busy = False
                return
            x1 = -1
            v = [dx, dy, 0]
            rot = pf.canvas.camera.rot[:3, :3]
            v2 = dot(v, linalg.inv(rot))
            V = v2/sqrt(dot(v2,v2.conj()))
            centers = cube.centroids()
            P1 = centers[element][0]
#            draw(Formex([[P1, P1+V]]), color=red, bbox='last', linewidth=3.0)
            planeAxis = argsort(abs(P1))[-1]
            pos = P1[planeAxis]>0
            rotateCube(cube, planeAxis, pos, P1, V)
        busy = False

def selectElement(self, x, y, w, h):
    """Returns the element closest to the cursor position"""
    self.selection.clear()
    self.selection.setType('element')
    self.selection_filter = None
    self.pick_window = (x,y,w,h,GL.glGetIntegerv(GL.GL_VIEWPORT))
    self.pick_parts('element', 54, store_closest=True)
    if len(self.picked) != 0:
        self.selection_front = self.closest_pick
        self.selection.set([self.closest_pick[0]])
    self.update()
    try:
        return self.selection[0]
    except:
        return [-1]

def rotateCube(self, planeAxis, pos, P, V, view=True):
    """Determine which elements should rotate in which direction."""
    tol = 0.001
    V[planeAxis] = 0.
    sorted = argsort(abs(V))
    rotAxis, dirAxis = int(sorted[1]), int(sorted[2])
    if (rotAxis+1) % 3 != dirAxis:
        dir = (pos==(V[dirAxis]>0))
    else:
        dir = not (pos==(V[dirAxis]>0))
    centers = self.centroids()
    rowElements = where(abs(centers[:, rotAxis]-P[rotAxis])<tol+0.5)[0]
    rotateRow(cube, rowElements, rotAxis, dir, steps, view)

def rotateRow(self, rowElements, rotAxis, dir, steps=1, view=True):
    """Rotate the rowElements around rotAxis in direction dir"""
    if dir:
        angle = 90.
    else:
        angle = -90.
    if view:
        global drawn
        if steps == 0:
            steps = 1
        for i in range(steps):
            self[rowElements] = self[rowElements].rotate(angle/steps, rotAxis)        
            dr = drawn
            drawn = draw(self, bbox='last')
            undraw(dr)
            time.sleep(t)
    else:
        self[rowElements] = self[rowElements].rotate(angle, rotAxis)

def perm(n=2):
    """Calulate the number of permutations for a nxnxn cube"""
    even = (n%2==0)
    if even:
        if n<3:
            return fac(8)*3**7/24
        else:
            return fac(8)*3**7*(fac(24)/fac(4)**6)**((n-2)/2)**2*fac(24)**((n-2)/2)/24
    else:
        if n<4:
            return fac(8)*3**7*fac(12)/2*2**11
        else:
            return fac(8)*3**7*(fac(24)/fac(4)**6)**(((n-3)/2)**2+(n-3)/2)*fac(12)/2*2**11*fac(24)**((n-3)/2)

def fac(x):
    """Return the factorial of x"""
    return reduce(lambda y,z:y*z,range(1,x+1))

# Dialogue
dia = None
def new():
    global cube, n, steps, t
    dia.acceptData()
    globals().update(dia.results)
    cube = createCube(n)
    refresh()

def set():
    global n,  steps, t
    dia.acceptData()
    res = dia.results
    if res['n']!=n:
        new()
    else:
        steps, t = res['steps'], res['t']

def randomize():
    global cube
    N = int(10*n + 5*n*rand())
    random = rand(N,6)
    centers = cube.centroids()
    for i in random:
        rotateCube(cube, int(3*i[0]), i[1]<0.5, centers[int(i[2]*6*n**2)], i[3:6], view=False)
    refresh()

def permutations():
    message('The total number of permutations of a %sx%s cube is:' % (n,n))
    N = str(perm(n))
    message(N)
    message('or roughly %s.%se%s' % (N[0],N[1:4],len(N)-1))

def close():
    pf.canvas.resetMouse(LEFT,SHIFT)
    dia.close()

def timeOut():
    show()
    close()

dia = widgets.NewInputDialog(
    caption='Cube',
    items=[
        I('n', 3,  text='Number of elements on a row', itemtype='slider', min=2, max=10, ticks=1),
        I('steps', 10, text='Animation steps', itemtype='slider', min=0, max=50, ticks=5),
        I('t', 0.05, text='Time between steps', min=0),  
   ],
    actions=[
        ('New',new),
        ('Set', set), 
        ('Shuffle',randomize),
        ('Permutations',permutations),
        ('Close',close),
    ])

if __name__ == "draw":
    clear()
    #renderMode('flatwire')
    cube = createCube()
    busy = False
    refresh()
    drawText('Hold SHIFT and move an element while pressing the LMB to move a row of the cube',20,40,color=blue,font='f',size=12)
    pf.canvas.setMouse(LEFT,turn,mod=SHIFT)
    dia.timeout = timeOut
    dia.show()
    dia.acceptData()
    globals().update(dia.results)

# End
