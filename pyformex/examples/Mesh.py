#!/usr/bin/env pyformex
# $Id$

"""Mesh

level = 'beginner'
topics = ['geometry', 'mesh']
techniques = ['extrude','border']

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

def showDescription():
    txt = __doc__.partition('.. Description')
    showText(' '.join(txt[1:]),modal=False)
    
clear()
view('iso')
smoothwire()
transparent()

showDescription()

n = 3,2,5
a = Formex(origin())

res = ask("Do you want a 2D or a 3D model?",["None","2D","3D"])
if res == "None":
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
print "%s elements" % m.nelems()

#export({'mesh':m})

pause('')
flat()
e = Mesh(m.coords,m.getEdges()).setProp(2)
clear()
draw(e)
drawNumbers(e)
print "%s edges" % e.nelems()

pause('')
e = Mesh(m.coords,m.getEdges(unique=True)).setProp(2)
clear()
draw(e)
drawNumbers(e)
print "%s edges" % e.nelems()

pause('')
smoothwire()
e = Mesh(m.coords,m.getFaces()).setProp(3)
clear()
draw(e)
drawNumbers(e)
print "%s faces" % e.nelems()

pause('')
e = Mesh(m.coords,m.getFaces(unique=True)).setProp(3)
clear()
draw(e)
drawNumbers(e)
print "%s faces" % e.nelems()

pause('')
e = Mesh(m.coords,m.getBorder()).setProp(4)
clear()
draw(e)
drawNumbers(e)
print "%s border elements" % e.nelems()

showInfo("Done")
