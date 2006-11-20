#!/usr/bin/env pyformex
# $Id: TrussFrame.py 151 2006-11-02 18:18:49Z bverheg $
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##

"""TrussFrame"""
clear()
yf = [ 0.0, 0.2, 1.2, 2.2, 3.2, 4.2, 4.5 ] # y of nodes in frame columns
a = Formex([[[0.0,y]] for y in yf ])
b = connect([a,a],bias=[0,1]).translate([0.5,0.0,0.0])
b.setProp(3)
c = b.reflect(0)
d = connect([b,c],bias=[1,1])
d.setProp(2)
e = connect([b,c],bias=[1,2]).select([0,2]) + connect([b,c],bias=[2,1]).select([1,3])
e.setProp(1)
col = b+c+d+e
frame = col.translate([-4.0,0.0,0.0]) + col.translate([+4.0,0.0,0.0])

# Dakligger
h0 = 1.2 # hoogte in het midden
h1 = 0.5 # hoogte aan het einde
xd = [ 0, 0.6 ] + [ 0.6+i*1.2 for i in range(5)] # hor. positie knopen
ko = Formex([[[x,0.0]] for x in xd])
ond = connect([ko,ko],bias=[0,1])
bov = ond.translate(1,h0).shear(1,0,(h1-h0)/xd[-1])
tss = connect([ond,bov],bias=[1,1])
ond.setProp(2)
bov.setProp(4)
tss.setProp(5)
dakligger = (ond+bov+tss)
dakligger += dakligger.reflect(0)
frame += dakligger.translate([0,yf[-1],0])
draw(frame)

structure = frame.replic2(2,6,12.,3.,0,2)
clear()
draw(structure)
view('top')
view('right')
view('iso')
