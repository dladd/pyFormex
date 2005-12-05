#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
#
"""TrussFrame"""
def show(F):
    clear()
    draw(F)
    

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
frame = col.translate([-4.0,0.0,0.0]) + col.translate([+4.0,0.0,0.0])

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
draw(frame)


structure = frame.replic2(2,6,0,2,12,3)
draw(structure)


view('top')
canvas.display()


view('right')
canvas.display()


view('iso')
canvas.display()
