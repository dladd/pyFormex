#!/usr/bin/env pyformex


"""I-profile"""

from plugins import sweep
from simple import circle, line

# GEOMETRICAL PARAMETERS FOR HE200B wide flange beam
h = 200. #beam height
b = 200. #flange width 
tf = 15. #flange thickness
tw = 9. #body thickness
l = 400. #beam length
r = 18. #filling radius

# MESH PARAMETERS
el = 20 #number of elements along the length
etb = 2 #number of elements over half of the thickness of the body
ehb = 5 #number of elements over half of the height of the body
etf = 5 #number of elements over the thickness of the flange
ewf = 8 #number of elements over half of the width of the flange
er = 6#number of elements in the circular segment

nbody,ebody = sweep.gridRectangle(etb,ehb,tw/2.,h/2.-tf-r)
Body = Formex(nbody[ebody].reshape(-1,4,3)).translate([0,tw/4.,h/4.-tf/2.-r/2.])

c1a = line([0,0,h/2-tf-r],[0,0,h/2-tf+tw/2],er/2)
c1b = line([0,0,h/2-tf+tw/2],[0,tw/2+r,h/2-tf+tw/2],er/2)
c1 = c1a + c1b
c2 = circle(90./er,90./er,90.).scale(r).rotate(-90.,1).rotate(90.,0).translate([0,tw/2+r,h/2-tf-r])
nfilled,efilled = sweep.gridBetween2Curves(c1,c2,etb)
Filled = Formex(nfilled[efilled].reshape(-1,4,3))

nflange1,eflange1 = sweep.gridRectangle(er/2,etf-etb,tw/2.+r,tf-tw/2.)
Flange1 = Formex(nflange1[eflange1].reshape(-1,4,3)).translate([0,tw/4.+r/2.,h/2.-(tf-tw/2.)/2.])

nflange2,eflange2 = sweep.gridRectangle(ewf,etf-etb,b/2.-r-tw/2.,tf-tw/2.)
Flange2 = Formex(nflange2[eflange2].reshape(-1,4,3)).translate([0,tw/2.+r+(b/2.-r-tw/2.)/2.,h/2.-(tf-tw/2.)/2.])

nflange3,eflange3 = sweep.gridRectangle(ewf,etb,b/2.-r-tw/2.,tw/2.)
Flange3 = Formex(nflange3[eflange3].reshape(-1,4,3)).translate([0,tw/2.+r+(b/2.-r-tw/2.)/2.,h/2.-tf+tw/4.])

Quarter = Body + Filled + Flange1 + Flange2 + Flange3
Half = Quarter.rosette(2,180.,2)
Beam = Half.rosette(2,180.,1)
nodesQuad,elemsQuad = Beam.feModel()
path = line([0,0,0],[0,0,l],el)
nodes,elems = sweep.sweepGrid(nodesQuad,elemsQuad,path,a1='last',a2='last')
draw(Formex(nodes[elems].reshape(-1,8,3)),eltype='hex',color='red',linewidth=2)


