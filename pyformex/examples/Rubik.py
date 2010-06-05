#!/usr/bin/env pyformex --gui
# $Id:$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

"""Rubik

level = 'normal'
topics = ['illustration']
techniques = ['colors','dialog','draw','persistence','random']

"""


"""Example: Rubik's cube
Note: inverse means counter clockwise!
Plane numbering: (0:Front,1:Back,2:Left,3:Right,4:Up,5:Down)

  |4|        |U|
|2|0|3|1|  |L|F|R|B|
  |5|        |D|
"""

expl = """How to solve a Rubik's cube:

(This is certainly not the fastest way, but it is rather easy to use and comprehend)

Scramble the cube either manually or with the "Randomize" button.

Step 1.
Create a cross at the bottom (white) plane, taking care that the sides are in the right direction
(meaning green and blue opposite to each other, as well as magenta and red)
This should be relatively easy to so with intuitions, so no algorithm is added.

Step 2.
Put the corners of the bottom plane at their right position and direction.
This should also be rather easy, so again no algorithm is provided.
When ready turn the middle squares of the side planes (magenta,blue,red,green) to their respective positions.

Step 3.
You should now have solved the bottom row.
For the middle row, turn the top row so that the top middle square is the same color as the middle square, and the color on the top plane is not the original top color (yellow if you didn't change the front yet).
This means that the top color has to turn left or right to get in the right position.
First set this plane to be the front plane, with "Set front".
Then use the algorithm "To left" or "To right" to turn it correctly.
Repeat this to position the four corner pieces of the middle row. Should at any point the needed piece be at the side in stead of at the top, use the same two algorithms to place it at the top.

Step 4.
You should now have solved the two lowest rows, only the top row should remain.
First, create a cross at the top plane by getting the middle squares facing upwards (do not look at their position for the moment, only at their orientation).
Use algorithm "To cross" for this.
If there is more than one side already at the right orientation, hold them in an --- or -' shape (by using U).
Do the "To cross" algorithm once in the first case, and twice in the second case.

Step 5.
Now the cross should be sorted in the right order.
If possible, use U to get more than one side square in the right position.
Use "To front" on a side plane to get the right squares in an I or L shape.
If only one can be brought to the right position, than the front plane can be chosen random.
Now use the "Sort cross" algorithm to get them to the correct position (once for L shape, twice for the others), until only the corners of the upper plane remain to be solved.

Step 6.
Get the corners to their position.
Use the "Sort corners" algorithm to get them into the right positions, with a correct one in the lower right corner of the upper plane.
You may have to use this a few times. Change the front plane if you have to in order to get a solved one in the lower right corner.

Step 7.
All parts are in the right spot now, but the upper corners need to be oriented.
For this you use the final algorithm "Orient corners".
Make sure that an incorrect one is always in the lower right corner of the upper plane.
It can be easier if you put the first correct one in  the lower left corner.
Use U to turn when a correctly oriented one arrives in the lower right corner (should each time be after using the algorithm either 2 two or four times).
Do not change the front plane while executing step 7!
Magically, once you orient the last corner, you only have to do U to solve the cube.

Your cube should be solved now, congratulations!
"""


from numpy import *
from formex import *
from plugins import formex_menu
from plugins.mesh import *
from numpy.random import rand
import gui.widgets
from gui.widgets import simpleInputItem as I
from gui.draw import showText

# Geometry generation
def createRubik():
    """Creates a new Rubik cube, with all colors at the right spot"""
    base = Formex(mpattern('123')).replic2(3,3,1,1)
    front = base.translate(2,3)
    back = base.rotate(180,1).translate([3,0,0])
    left = base.rotate(-90,1)
    right = base.rotate(90,1).translate([3,0,3])
    up = base.rotate(-90,0).translate([0,3,3])
    down = base.rotate(90,0)
    F = front + back + left + right + up + down
    F = F.translate([-1.5,-1.5,-1.5])
    prop = ones((6,3,3),int) #Front = red
    prop[1] = 5 * prop[1]    #Back = magenta
    prop[2] = 3 * prop[2]    #Left = blue
    prop[3] = 2 * prop[3]    #Right = green
    prop[4] = 6 * prop[4]    #Up = yellow
    prop[5] = 7 * prop[5]    #Down = white
    setProp(F,prop)
    return F

# Drawing definitions
def refresh(self,undo=False):
    """Refresh the view of the cube"""
    clear()
    draw(self)
    #drawNumbers(self)
    #drawText('generated by pyFormex',10,20,color=blue,font='f',size=12)
    if not undo:
        global undoList
        undoList = append(undoList[1:],getProp(self)).reshape(-1,6,3,3)

# General property changing
def setProp(self,prop):
    """Apply the new color properties to the cube"""
    self.prop = prop.ravel()

def getProp(self):
    """Returns the cube's color in a 6x3x3 matrix notation"""
    return self.prop.reshape(6,3,3)

def change(self,plane):
    """This is just a development function to assign colors to certain squares of the cube"""
    oldprop = self.prop.reshape(6,9)
    prop = oldprop.copy()
    #for i in range(9):
        #prop[plane,i] = i
    prop[plane,0] = 0
    self.prop = prop.ravel()
    return self

# General rotations
def rot(prop,inv=False):
    """Return the new colors of the rotating plane. inv stands for inverted (i.e. counter clockwise) rotation"""
    if inv:
        return column_stack((prop[2][:,newaxis],prop[1][:,newaxis],prop[0][:,newaxis]))
    else:
        return column_stack((prop[0][:,newaxis][::-1],prop[1][:,newaxis][::-1],prop[2][:,newaxis][::-1]))

def rotnb(oldprop,plane,inv=False):
    """Return the new colors of the neighbouring planes when the plane is rotated. inv = counter clockwise"""
    nb,pos = neighbours[plane],positions[plane]
    prop = oldprop.copy()
    a = -1
    if inv:
        a = 1
    for i in range(4):
        j = (i+a)%4
        if pos[i][0]==-1:
            if pos[i][2]==-1:
                prop[nb[i],:,pos[i][1]] = cut(oldprop,nb[j],pos[j])
            else:
                prop[nb[i],:,pos[i][1]][::-1] = cut(oldprop,nb[j],pos[j])
        else:
            if pos[i][2]==-1:
                prop[nb[i],pos[i][0],:] = cut(oldprop,nb[j],pos[j])
            else:
                prop[nb[i],pos[i][0],:][::-1] = cut(oldprop,nb[j],pos[j])
    return prop

def cut(prop,nb,pos):
    """A helping function for rotnb."""
    if pos[0]==-1:
        tmp = prop[nb,:,pos[1]]
    else:
        tmp = prop[nb,pos[0],:]
    if pos[2]==-1:
        return tmp
    else:
        return tmp[::-1]

# Plane rotations
def rotatePlane(self,plane=0,inv=False):
    """Rotate a certain plane."""
    oldprop = getProp(self)
    prop = oldprop.copy()
    prop[plane] = rot(oldprop[plane],inv)
    prop = rotnb(prop,plane,inv)
    setProp(self,prop)
    return self

# Short expressions
def F(self):
    return rotatePlane(self,0)

def Fi(self):
    return rotatePlane(self,0,True)

def B(self):
    return rotatePlane(self,1)

def Bi(self):
    return rotatePlane(self,1,True)

def L(self):
    return rotatePlane(self,2)

def Li(self):
    return rotatePlane(self,2,True)

def R(self):
    return rotatePlane(self,3)

def Ri(self):
    return rotatePlane(self,3,True)

def U(self):
    return rotatePlane(self,4)

def Ui(self):
    return rotatePlane(self,4,True)

def D(self):
    return rotatePlane(self,5)

def Di(self):
    return rotatePlane(self,5,True)

# Some algorithms that can be used to solve the cube
# Note that you should read the algorithms from right to left, consistent with nested functions
def alg1(self):
    """Algorithm for step 3: top block to the right"""
    return F(U(Fi(Ui(Ri(Ui(R(U(self))))))))

def alg2(self):
    """Algorithm for step 3: top block to the left"""
    return Fi(Ui(F(U(L(U(Li(Ui(self))))))))

def alg3(self):
    """Algorithm for step 4: creating a cross at the top"""
    return Fi(Ui(Ri(U(R(F(self))))))

def alg4(self):
    """Algorithm for step 5: sorting the cross at the top"""
    return Ri(U(U(R(U(Ri(U(R(self))))))))

def alg5(self):
    """Algorithm for step 6: corners at the top to their right position, but not necessarily the right direction"""
    return L(Ui(Ri(U(Li(Ui(R(U(self))))))))

def alg6(self):
    """Algorithm for step 7: corners at the top rotating to the right direction"""
    return D(R(Di(Ri(self))))

# Main program
clear()
#GD.canvas.setBgColor('#333366','#acacc0')
#GD.GUI.drawwait = 1.
renderMode('flatwire')
view('iso')
#toolbar.setProjection()
l,r,b,t = [-1,0,-1],[-1,2,-1],[0,-1,-1],[2,-1,-1]
li,ri,bi,ti = [-1,0,0],[-1,2,0],[0,-1,0],[2,-1,0]
global neighbours # List of the neighbouring planes, clockwise starting from right
neighbours = [[3,5,2,4],[2,5,3,4],[0,5,1,4],[1,5,0,4],[3,0,2,1],[3,1,2,0]]
global positions
positions = [[li,ti,r,b],[l,bi,ri,t],[l,l,ri,l],[li,r,r,r],[t,t,t,t],[b,b,b,b]]
choices = ['F','B','L','R','U','D']
global undoList
buf=10
undoList = ndarray((buf,6,3,3)).astype(int)
global rub
rub = createRubik()
#rub = change(rub,2)
refresh(rub)

# Dialog and its functions
dia = None

def close():
    """Close the dialog"""
    dia.close()

def rotateF():
    execRotateGeneral(0)

def rotateB():
    execRotateGeneral(1)

def rotateL():
    execRotateGeneral(2)

def rotateR():
    execRotateGeneral(3)

def rotateU():
    execRotateGeneral(4)

def rotateD():
    execRotateGeneral(5)

def rotateFi():
    execRotateGeneral(0,True)

def rotateBi():
    execRotateGeneral(1,True)

def rotateLi():
    execRotateGeneral(2,True)

def rotateRi():
    execRotateGeneral(3,True)

def rotateUi():
    execRotateGeneral(4,True)

def rotateDi():
    execRotateGeneral(5,True)

def execRotateGeneral(plane=0,inv=False):
    global rub
    rub = rotatePlane(rub,plane,inv)
    refresh(rub)

def algorithm1():
    global rub
    rub = alg1(rub)
    refresh(rub)
    message('Algorithm: U R Ui Ri Ui Fi U F')

def algorithm2():
    global rub
    rub = alg2(rub)
    refresh(rub)
    message('Algorithm: Ui Li U L U F Ui Fi')

def algorithm3():
    global rub
    rub = alg3(rub)
    refresh(rub)
    message('Algorithm: F R U Ri Ui Fi')

def algorithm4():
    global rub
    rub = alg4(rub)
    refresh(rub)
    message('Algorithm: R U Ri U R U U Ri')

def algorithm5():
    global rub
    rub = alg5(rub)
    refresh(rub)
    message('Algorithm: U R Ui Li U Ri Ui L')

def algorithm6():
    global rub
    rub = alg6(rub)
    refresh(rub)
    message('Algorithm: Ri Di R D')

def howToSolve():
    """Shows the steps you can follow to easily solve the cube"""
    showText(expl,modal=False)

def new():
    """Create a new (solved) cube"""
    global rub
    rub = createRubik()
    refresh(rub)

def randomize():
    """Apply random rotations to the cube, i.e. scramble the cube"""
    global rub
    step = [' F',' B',' L',' R',' U',' D']
    n = int(10 + 10*rand()) # Number of random rotations
    random = rand(n,2)
    steps = ''
    for i in random:
        rr = int(6*i[0])
        steps += step[rr]
        inv = i[1]<0.5
        if inv:
            steps += 'i'
        rub = rotatePlane(rub,rr,inv)
    message('The apllied random rotations are:%s' % steps)
    refresh(rub)

def undo():
    """Undo the last moves so that the cube gets back into the last viewed shape"""
    global rub
    global undoList
    last = undoList[-2]
    if sum(last)==0:
        warning('Maximum number of undo\'s reached!')
        return
    rub.prop = last.ravel()
    undoList = append(zeros((6,3,3),int),undoList[:-1]).reshape(-1,6,3,3)
    refresh(rub,True)

def setFront():
    """Select a plane that you would like to have as the front plane (rotates the whole cube)"""
    global rub
    planerotations = [[0,0],[0,180],[0,-90],[0,90],[-90,0],[90,0]]
    p = pick('element',filtr='single')[0]
    pl = planerotations[int(p/9)]
    propold = rub.prop
    rubtmp = rub.rotate(pl[0],0).rotate(pl[1],1)
    cent = rub.centroids()
    centnew = rubtmp.centroids()
    tmp = map(lambda x:x[0]+x[1]*10.+x[2]*100.,cent)
    tmp2 = map(lambda x:x[0]+x[1]*10.+x[2]*100.,centnew)
    newnumbers = asarray(map(lambda x: where(tmp==x)[0],tmp2)).ravel()
    rub.prop = propold[newnumbers]
    refresh(rub)
    view('front')

def timeOut():
    show()
    close()

b1 = widgets.ButtonBox('Basic',actions=[('F',rotateF),('B',rotateB),('L',rotateL),('R',rotateR),('U',rotateU),('D',rotateD)])
b2 = widgets.ButtonBox('Inverse',actions=[('Fi',rotateFi),('Bi',rotateBi),('Li',rotateLi),('Ri',rotateRi),('Ui',rotateUi),('Di',rotateDi)])
b3 = widgets.ButtonBox('Algorithms',actions=[('To right',algorithm1),('To left',algorithm2),('To cross',algorithm3),('Sort cross',algorithm4),('Sort corners',algorithm5),('Orient corners',algorithm6)])
b4 = widgets.ButtonBox('',actions=[('How to solve the cube',howToSolve)])

data_items1 = [b1,b2,b3]

data_items2 = [b4]

dia = widgets.NewInputDialog(
    caption="Rubik's Cube",
    items=[
        ('Cube transformations',data_items1),
#        ('Help',data_items2),
    ],
    actions=[
        ('How to solve the cube',howToSolve),
        ('New',new),
        ('Randomize',randomize),
        ('Undo',undo),
        ('Set front',setFront),
        ('Close',close),
    ])

dia.timeout = timeOut
#dia.resize(800,200)
dia.show()
