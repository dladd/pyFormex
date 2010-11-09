#!/usr/bin/env pyformex
# $Id$

"""RotoTranslation

level = 'advanced'
topics = ['geometry']
techniques = ['transform']
author: gianluca

.. Description

RotoTranslation
---------------
This example illustrates the use of rototranslate() to return to an original
reference system after a number of affine transformations.
"""
from numpy import *
from plugins.geomtools import rototranslate

def drawCS(ax=None):
    """it draws the coordinate system (x,y,z,o)"""
    if ax==None:ax=Coords([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.], [0., 0., 0.]])
    draw(Formex(ax[[3, 0, 3, 1, 3, 2]].reshape(-1, 2, 3)).setProp([1, 2, 3]), linewidth=2)
    draw(Formex(ax[:3]).setProp([1, 2, 3]))
    #drawNumbers(Formex(ax[:3]))



def refCS():
    return Coords([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.], [0., 0., 0.]])

clear()
lights(True)
view('iso')
smooth()
transparent(state=False)

chdir(pf.cfg['curfile'])
F = Formex.read(pf.cfg['pyformexdir']+'/data/horse.pgf')#file with horse stl
F = F.toMesh().translate(-F.center())
xmin,xmax = F.bbox()
M0 = F.scale(1./(xmax[0]-xmin[0])).rotate(180, 1)

#bind a Coords (Formex or Mesh) to its reference system
M0
cs0=refCS()

#draw
dM0=draw(M0, color='white')
drawCS(cs0)

tx='There is a white free horse.'
drawText(tx, 100, 250, size=15)
zoomAll()
sleep(3)
tx='A bad pyFormex user puts the horse in a cage and transports it around. The angry horse changes colour at each step.'


#same combination of rotations / translations applied F0 and cs0 to obtain F1 and cs1
M1=M0.translate([-4, -3, 0.]).setProp(0)
cs1=cs0.translate([-4,-3, 0.])

D=[]
for tf in range(1, 18, 1):
    D.append(drawBbox(M1))
    drawCS(cs1)
    D.append(draw(M1.setProp(tf), bbox=array([[-20., -20., -20.], [20., 20., 20.]])))
    if tf==1:
        drawText(tx, 100, 200, size=15)
        zoomAll()
        sleep(5)
    M1=M1.rotate(30, 1).rotate(-10., 2).translate([0., -tf*0.1, 0.])
    cs1=cs1.rotate(30, 1).rotate(-10., 2).translate([0., -tf*0.1, 0.])
    sleep(0.1)

zoomAll()
sleep(1)

tx='Finally the horse escapes from the cage and wants to go back home. If it stays black, it cannot be seen.'
drawText(tx, 100, 150, size=15)


M1=M1.setProp(0)
c1=M1.center()
draw(M1, bbox=array([[-20., -20., -20.], [20., 20., 20.]]))

zoomAll()



sleep(4)
tx='It forgot the way back !!!'
drawText(tx, 100, 100, size=15)
sleep(1)
for d in D:
    undraw(d)
    sleep(0.1)
    zoomAll()

tx='Thanks to its "orientation" can go back in one single step, crossing the bushes.'
drawText(tx, 100, 50, size=15)

#go back in 1 step to the original sys0
#sleep(1)
n1, e1=M1.coords, M1.elems
n2=rototranslate(n1, cs1,cs0)
M2=M1._set_coords(Coords(n2))


c2=M2.center()
sleep(4)
draw(Formex([[c1, c2]]), linewidth=3)

draw(M2)
drawCS(cs0)
zoomAll()

for i in range(6):
    transparent(state=i%2)
    sleep(0.5)













