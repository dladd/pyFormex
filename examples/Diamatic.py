#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
"""Double layer diamatic dome"""
clear()
global out
rt = 36.   # radius of top circumsphere
rb = 34.5  # radius of bottom circumsphere
m = 6      # frequency of top layer
n = 6      # number of sectors
a = 36.    # sweep angle of top layer
T = Formex([[[rt,0,0],[rt,0,3]],[[rt,0,0],[rt,3,3]],[[rt,0,3],[rt,3,3]]])
W1 = Formex([[[rb,1,2],[rt,0,0]],[[rb,1,2],[rt,0,3]],[[rb,1,2],[rt,3,3]]])
W2 = Formex([[[rb,2,4],[rt,3,6]],[[rb,2,4],[rt,0,3]],[[rb,2,4],[rt,3,3]]])
B1 = Formex([[[rb,2,4],[rb,1,2]],[[rb,2,4],[rb,1,5]],[[rb,2,4],[rb,4,5]]])
B2 = Formex([[[rb,1,2],[rb,-1,2]]])
top = T.genit(1,m,3,3,0,1)
web = W1.genit(1,m,3,3,0,1) + W2.genit(1,m-1,3,3,0,1)
bot = B1.genit(1,m-1,3,3,0,1) + B2.rin(3,m,3)
F = (top+web+bot)#.spherical(1.,360./n,a/(3*m))

def toSector(f):
    x,y,z = f.x(), f.y(), f.z()
    d = sqrt(y*y+z*z)
    return f.map(lambda x,y,z:[x,where(d>0,y*z/d,0),where(d>0,z*z/d,0)])

#out = toSector(top)#.rosette(4,0,[0,0,0],45.)
out = top + toSector(top)
draw(out)
