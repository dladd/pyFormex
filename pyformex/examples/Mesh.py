#!/usr/bin/env pyformex
# $Id$
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

"""Mesh

level = 'beginner'
topics = ['geometry', 'mesh']
techniques = ['extrude','border','pause']

.. Description

Mesh
----
This example illustrates some of the powerful methods provided by the
**Mesh** class. The example constructs a 2D or 3D (selectable by the user)
mesh by extruding a point into a line, the line into a surface, and in case
of a 3D structure, the surface into a volume.

In a sequence of steps, the example then draws the following geometries with
element numbers included:

- the extruded geometry,
- all the edges of all elments,
- the unique edges,
- all the faces of all elements,
- the unique faces,
- the border (in the 3D case, this is a set of faces, in the 2D case it is a
  set of lines.
  
At each step the number of elements is printed.

Remark
------
The script pauses at each step.
The user should click the **STEP** button to move the the next step, or the
**CONTINUE** button to move to the end of the script.

Script
------
Notice how the same script with the same methods is working for both the 2D
and the 3D cases.

Exercises
---------
1. Make this script also work for the 1D case. 

"""
def atExit():
    #print "THIS IS THE EXIT FUNC"
    GD.GUI.setBusy(False)
    


GD.GUI.setBusy()
#draw.logfile = open('pyformex.log','w')
    
clear()
view('iso')
smoothwire()
transparent()


n = 3,2,5
a = Formex(origin())

res = ask("Choose the model:",["None","2D","3D","Help"])
if res == "None":
    exit()
if res == "Help":
    showDescription()
    exit()

ndim = int(res[0])

for i in range(ndim):
    a = a.extrude(n[i],1.,i)
    
draw(a)
drawNumbers(a)

clear()
m = a.toMesh().setProp(1)
draw(m)
drawNumbers(m)
message("%s elements" % m.nelems())

export({'mesh':m})

pause('')
flat()
e = Mesh(m.coords,m.getEdges()).setProp(2)
clear()
draw(e)
drawNumbers(e)
message("%s edges" % e.nelems())

pause('')
e = Mesh(m.coords,m.getEdges(unique=True)).setProp(2)
clear()
draw(e)
drawNumbers(e)
message("%s unique edges" % e.nelems())

pause('')
smoothwire()
e = Mesh(m.coords,m.getFaces()).setProp(3)
clear()
draw(e)
drawNumbers(e)
message("%s faces" % e.nelems())

pause('')
e = Mesh(m.coords,m.getFaces(unique=True)).setProp(3)
clear()
draw(e)
drawNumbers(e)
message("%s unique faces" % e.nelems())

pause('')
e = Mesh(m.coords,m.getBorder()).setProp(4)
clear()
draw(e)
drawNumbers(e)
message("%s border elements" % e.nelems())


export({'border':e})

showInfo("Done")
