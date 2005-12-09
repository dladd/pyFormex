#!/usr/bin/env pyformex
# $Id$
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
