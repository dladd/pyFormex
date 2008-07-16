#!/usr/bin/env pyformex --gui
# $Id$

import simple

nx,ny = 8,6

F = simple.rectangle(nx,ny)
F = F.trl(-F.center()+[0.,0.,1.])
draw(F)

x = F.f.projectOnSphere(ny)
G = Formex(x)
draw(G,color=red)

x = F.f.rotate(30).projectOnCylinder(ny)
H = Formex(x)
draw(H,color=blue)

#End
