#!/usr/bin/env pyformex
# $Id$
#
"""TrussFrame"""
def show(F):
    clear()
    drawProp(F)
    sleep()

yf = [ 0.0, 0.2, 1.2, 2.2, 3.2, 4.2, 4.5 ] # y of nodes in frame columns
a = Formex([[[0.0,y]] for y in yf ])
b = Formex.connect([a,a],bias=[0,1]).translate([0.5,0.0,0.0])
b.setProp(3)
c = b.reflect(0)
d = Formex.connect([b,c],bias=[1,1])
d.setProp(2)
e = Formex.connect([b,c],bias=[1,2]).select([0,2]) + Formex.connect([b,c],bias=[2,1]).select([1,3])
e.setProp(1)
col = b+c+d+e
frame = col.translate([-4.0,0.0,0.0]) +  col.translate([+4.0,0.0,0.0])

# Dakligger
h0 = 1.2 # hoogte in het midden
h1 = 0.5 # hoogte aan het einde
xd = [ 0, 0.6 ] + [ 0.6+i*1.2 for i in range(5)] # hor. positie knopen
ko = Formex([[[x,0.0]] for x in xd])
ond = Formex.connect([ko,ko],bias=[0,1])
bov = ond.translate1(1,h0).shear(1,0,(h1-h0)/xd[-1])
tss = Formex.connect([ond,bov],bias=[1,1])
ond.setProp(2)
bov.setProp(4)
tss.setProp(5)
dakligger = (ond+bov+tss)
dakligger += dakligger.reflect(0)
frame += dakligger.translate([0,yf[-1],0])
clear()
drawProp(frame)
sleep()

structure = frame.generate2(2,6,0,2,12,3)
drawProp(structure)

sleep()
canvas.camera.rotate(90,1)
canvas.display()

sleep()
canvas.camera.rotate(-90,0)
canvas.display()

sleep()
canvas.camera.rotate(-90,1)
canvas.display()

sleep()
canvas.camera.rotate(-90,2)
canvas.display()
