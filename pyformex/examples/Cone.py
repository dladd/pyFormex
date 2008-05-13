#!/usr/bin/env python pyformex.py
# $Id$

import simple

smooth()
r=3.
h=15.
n=64

F = simple.sector(r,360.,n,n,h=h,diag=None)
F.setProp(0)
draw(F,view='bottom')
zoomall()


ans = ask('How many balls do you want?',['3','2','1','0'])
nb = int(ans)

if nb > 0:
    B = simple.sphere3(n,n,r=0.9*r,bot=-90,top=90)
    B1 = B.translate([0.,0.,0.95*h])
    B1.setProp(1)
    draw(B1,bbox=None)
    zoomall()

if nb > 1:
    B2 = B.translate([0.2*r,0.,1.15*h])
    B2.setProp(2)
    draw(B2,bbox=None)
    zoomall()

if nb > 2:
    B3 = B.translate([-0.2*r,0.1*r,1.25*h])
    B3.setProp(6)
    draw(B3,bbox=None)
    zoomall()
