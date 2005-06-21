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
u = 3.     # modular length
rt = 36.   # radius of top circumsphere
rb = 34.5  # radius of bottom circumsphere
m = 6      # frequency of top layer
n = 6      # number of sectors
a = 36.    # sweep angle of top layer
# We start with a modular length = 3, for convenience
T = Formex([[[0,0],[3,0]],[[3,0],[3,3]],[[3,3],[0,0]]],3)
B1 = Formex([[[4,2],[2,1]],[[4,2],[5,1]],[[4,2],[5,4]]],0)
B2 = Formex([[[2,1],[2,-1]]],0)
#W1 = Formex([[[1,2,rb],[0,0,rt]],[[1,2,rb],[0,3,rt]],[[1,2,rb],[3,3,rt]]],1)
#W2 = Formex([[[2,4,rb],[3,6,rt]],[[2,4,rb],[0,3,rt]],[[2,4,rb],[3,3,rt]]],1)
W1 = Formex([[[1,2,rb],[0,0,rt]],[[1,2,rb],[0,3,rt]],[[1,2,rb],[3,3,rt]]],1)
W2 = Formex([[[2,4,rb],[3,6,rt]],[[2,4,rb],[0,3,rt]],[[2,4,rb],[3,3,rt]]],1)
#
top = T.replic2(m,m,3,3,0,1,u,-1)
web = W1.replic2(1,m,3,3,0,1) + W2.replic2(1,m-1,3,3,0,1)
bot = B1.replic2(m-1,m-1,u,u,0,1,u,-1) + B2.replic(m,3,0)
draw(T+B1+B2)
# Scale the parts and circulize them
#for F in [ top,bot]:
#    F.scale(u/3.).circulize()
#sleep()
#clear()
#draw(top+bot)
#top.mapd(2,lambda d:sqrt(rt**2-d**2),[0,0,0],[0,1])
#bot.mapd(2,lambda d:sqrt(rb**2-d**2),[0,0,0],[0,1])
#sleep()
#clear()
#draw(top+bot)
#dome=(top+bot).rosette(6,60)
#sleep()
#clear()
#draw(dome)
