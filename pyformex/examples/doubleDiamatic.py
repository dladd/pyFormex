#!/usr/bin/env pyformex --gui
# $Id: DoubleDiamatic.py 53 2005-12-05 18:23:28Z bverheg $
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
#
"""Double layer diamatic dome ** It works ok? **

level = 'beginner'
topics = ['geometry','domes']
techniques = []

"""

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
#
#clear()
#draw(top+bot)
#top.mapd(2,lambda d:sqrt(rt**2-d**2),[0,0,0],[0,1])
#bot.mapd(2,lambda d:sqrt(rb**2-d**2),[0,0,0],[0,1])
#
#clear()
#draw(top+bot)
#dome=(top+bot).rosette(6,60)
#
#clear()
#draw(dome)
