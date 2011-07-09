#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""RotoTranslation

level = 'advanced'
topics = ['geometry']
techniques = ['transform']
original idea: gianluca

.. Description

RotoTranslation
---------------
This example illustrates the use of transformCS() to return to an original
reference system after a number of affine transformations.
"""
import simple

savewait = pf.GUI.drawwait
pf.GUI.drawwait = 2.0

def atExit():
    pf.GUI.drawwait = savewait

line = 300
line_inc = -40

def createScene(text=None,caged=True,color=None,move=0):
    """Create a scene of the story.

    The scene draws the horse (H), with the specified color number (0..7),
    caged or not, with the local axes (CS), and possibly a text.
    If move > 0, the horse moves before the scene is drawn.
    The horse and cage actors are returned.
    """
    global line,H,C,CS
    if move:
        H,C,CS = [ i.rotate(30,1).rotate(-10.,2).translate([0.,-move*0.1,0.]) for i in [H,C,CS] ]

    if caged:
        cage = draw(C,mode='wireframe',wait=False,)
    else:
        cage = None
    if color is None:
        color = 1 + random.randint(6)
    H.setProp(color)
    horse = draw(H)
    if text:
        drawText(text,20,line,size=20)
        line += line_inc
    axes = drawAxes(CS)
    zoomAll()
    return horse,cage


clear()
lights(True)
view('iso')
smooth()
transparent(state=False)
linewidth(2)
setDrawOptions({'bbox':None})

# read the model of the horse
F = Formex.read(getcfg('datadir')+'/horse.pgf')
# make sure it is centered
F = F.centered()
# scale it to unity size and head it in the x-direction
xmin,xmax = F.bbox()
H = F.scale(1./(xmax[0]-xmin[0])).rotate(180, 1)
# create the global coordinate system
CS0 = CS = CoordinateSystem()
# some text

# A storage for the scenes
script = []

# Scene 0: The story starts idyllic
T = 'There once was a white horse running free in the wood.'
script += [ createScene(text=T,caged=False,color=7) ]
pause()

# Scene 1: Things turn out badly
T = 'Some wicked pyFormex user caged the horse and transported it around.'
# apply same transformations on model and coordinate system
H,CS = [ i.translate([0.,3.,6.]) for i in [H,CS] ]
C = simple.cuboid(*H.bbox())
script += [ createScene(text=T) ]
pause()

# Scene 2..n: caged movements
T = 'The angry horse randomly changed colour at each step.'
script += [ createScene(text=T,move=1) ]
m = len(script)
n = 16
script += [ createScene(move=i) for i in range(m,n,1) ]
pause()


# Scene n+1: the escape
T = 'Finally the horse managed to escape from the cage.\nIt wanted to go back home and turned black, so it would not be seen in the night.'
escape = script[-1]
script += [ createScene(text=T,color=0,caged=False) ]
undraw(escape)
pause()

# The problem
T = "But alas, it couldn't remember how it got there!!!"
drawText(T,20,line,size=20)
line += line_inc
for s in script[:-2]:
    #sleep(0.1)
    undraw(s)
pause()

# The solution
T = "But thanks to pyFormex's orientation,\nit could go back in a single step, straight through the bushes."
drawText(T,20,line,size=20)
line += line_inc
H = H.transformCS(CS0,CS)
draw(Formex([[CS[3],CS0[3]]]))
pause()

T = "And the horse lived happily ever after."
script += [ createScene(text=T,color=7,caged=False) ]
undraw(script[-2])

# End











