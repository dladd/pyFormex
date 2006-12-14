#!/usr/bin/env pyformex
# $Id: SpaceTrussRoof.py 66 2006-02-20 20:08:47Z bverheg $
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Double Layer Flat Space Truss Roof"""
dx = 180 # Modular size (cm)
ht = 150 # Deck height
nx = 14  # number of bottom deck modules in x direction (should be even)
ny = 14  # number of bottom deck modules in y direction (should be even)
colht = 560  # Column height
m = 2        # Column multiplicity: should be an integer divisor of nx and ny
coldx = m * dx # column distance (should be a multiple of dx)
ncx = nx/m + 1  # number of columns in x-direction
ncy = ny/m + 1  # and in y-direction


bot = (Formex("1").replic2(nx,ny+1,1,1) + Formex("2").replic2(nx+1,ny,1,1)).scale(dx)
bot.setProp(3)
top = (Formex("1").replic2(nx+1,ny+2,1,1) + Formex("2").replic2(nx+2,ny+1,1,1)).scale(dx).translate([-dx/2,-dx/2,ht])
top.setProp(0)
T0 = Formex(4*[[[0,0,0]]]) # 4 times the corner of the bottom deck
T4 = top.select([0,1,nx+1,nx+2]) # 4 nodes of corner module of top deck
dia = connect([T0,T4]).replic2(nx+1,ny+1,dx,dx)
dia.setProp(1)
col = (Formex([[[0,0,-colht],[0,0,0]]]).replic2(ncx,2,m,ny) + Formex([[[0,m,-colht],[0,m,0]]]).replic2(2,ncy-2,nx,m)).scale([dx,dx,1])
col.setProp(2)

F = top+bot+dia+col
clear()
linewidth(1)
draw(F)

F = F.rotate(-90,0) # put the structure upright
clear()
draw(F)

setview('myview1',(30.,0.,0.))
view('myview1',True)

drawtimeout = 1
for i in range(19):
    setview('myview2',(i*10.,20.,0.))
    view('myview2',True)

# fly tru
if ack("Do you want to fly through the structure?"):
    totaltime = 10
    nsteps = 20
    # make sure bottom iz at y=0
    F = F.translate(1,-F.bbox()[0,1])
    clear()
    linewidth(1)
    draw(F)
    bb = F.bbox()
    # Fly at reasonable height
    bb[0,1] = 0.25 * bb[1,1]
    bb[1,1] = 0.75 * bb[1,1]
    ends = interpolate(Formex([[bb[0]]]),Formex([[bb[1]]]),[-0.5,0.8])
    path = divide(connect([ends,ends],bias=[0,1]),nsteps)
    linewidth(2)
    draw(path)
    steptime = float(totaltime)/nsteps
    flyAlong(path,sleeptime=steptime)
