#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Scallop Dome"""
# This example is fully annotated with comments in the statusbar
# First we define a function to display a Formex and then wait for the user
# to click the Step button
def show(F,view='front',clearscr=True):
    if clearscr:
        clear()
    draw(F,view)
    GD.canvas.update()
    
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
f2.setProp(2)
show(f1+f2)
#
message("Make circular copies to obtain a full circle")
show(f1+f2.rosette(6,60.))
# Create and display a scallop dome using the following parameters:
# n = number of repetitions of the base module in circumference (this does not
#     have to be equal to 6: the base module will be compressed/expanded to
#     generate a full circle
# f = if 1, the dome will have sharp edges where repeated modules meet;
#     if 2, the dome surface will be smooth over neighbouring modules.
# c = height of the dome at the center of the dome.
# r = height of the arcs at the circumference of the dome. 
def scallop(n,f,c,r):
    func = lambda x,y,z: [x,y,c*(1.-x*x/64.)+r*x*x/64.*4*power((1.-y)*y,f)]
    a=360./n
    f3 = f2.toCylindrical([0,1,2]).scale([1.,1./60.,1.])
    f4 = f3.map(func).cylindrical([0,1,2],[1.,a,1.]).rosette(n,a)
    message("Scallop Dome with n=%d, f=%d, c=%f, r=%f" % (n,f,c,r))
    return f4

message("Create a dome from the circular layout")
f2.setProp(3)
GD.canvas.camera.setRotation(0,-45)
show(scallop(6,1,2,0),0,False)


howmany = ask("How many / Which domes do you want?",
              ['One','Sequence','Custom','None'])

n,f,c,r = [6,1,2.,0.]

if howmany == 'One':
   # The example from the pyformex homepage
   show(scallop(6,1,4,4),0)
        
elif howmany == 'Sequence':
   # Present some nice examples
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
        show(scallop(n,f,c,r),0)

elif howmany == 'Custom':
   # Customized version
   while True:
       res = askItems([['n',n],['f',f],['c',c],['r',r]])
       n = int(res['n'])
       f = int(res['f'])
       c = float(res['c'])
       r = float(res['r'])
       show(scallop(n,f,c,r),0)
       if not ack("Want to try another one?"):
           exit()


       
   
