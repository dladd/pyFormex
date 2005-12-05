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
"""Scallop Dome"""
# This example is fully annotated with comments in the statusbar
# First we define a function to display a Formex and then wait for the user
# to click the Step button
def show(F,side='front'):
    clear()
    draw(F,side)
    
# Here we go
message("Create a triangular pattern in the first octant")
f1 = Formex([[[0,0],[1,0]],[[1,0],[1,1]]]).replic2(8,8,1,1,0,1,1,-1) + Formex([[[1,0],[2,1]]]).replic2(7,7,1,1,0,1,1,-1)
show(f1)
#
message("Remove some of the bars")
f1 = f1.remove(Formex([[[1,0],[1,1]]]).replic(4,2,0))
show(f1)
#
message("Transform the octant into a circular sector")
f2 = f1.circulize1()
f1.setProp(1)
f2.setProp(0)
show(f1+f2)
#
message("Make circular copies to obtain a full circle")
show(f1+f2.rosette(6,60.))
# Create and display a scallop dome using the following parameters:
# n = number of repetitions of the base module in circumference (this does not
#     have to be equal to 6: the base module will be compressed/expanded to
#     generate a full circle
# f = if 0, the dome will have sharp edges where repeated mdules meet;
#     if 1, the dome surface will be smooth over neighbouring modules.
# c = height of the dome at the center of the dome.
# r = height of the arcs at the circumference of the dome. 
def scallop(n,f,c,r):
    func = lambda x,y,z: [x,y,c*(1.-x*x/64.)+r*x*x/64.*4*power((1.-y)*y,f)]
    a=360./n
    f3 = f2.toCylindrical([0,1,2]).scale([1.,1./60.,1.])
    f4 = f3.map(func).cylindrical([0,1,2],[1.,a,1.]).rosette(n,a)
    message("Scallop Dome with n=%d, f=%d, c=%f, r=%f" % (n,f,c,r))
    show(f4,0)
#    return f4

# Present some nice examples
canvas.camera.setRotation(0,-45)
for n,f,c,r in [
    [6,1,2,0],
    [6,1,2,2],
    [6,1,2,5],
    [6,1,2,-2],
    [6,1,-4,4],
    [6,1,0,4],
    [6,1,4,4],
    [6,2,2,-4],
    [6,2,2,4],
    [6,2,2,8],
    [12,1,2,-2],
    [12,1,2,2] ]:
    scallop(n,f,c,r)

# That's all
#F=scallop(6,1,4,4)
#F.setProp(3)
#clear()
#draw(F)
