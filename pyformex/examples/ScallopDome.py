#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Scallop Dome

level = 'normal'
topics = ['geometry','domes']
techniques = ['dialog', 'color']

"""
#pf.canvas.settings['colormap'][2] = [1.,0.3,0.]

# This example is fully annotated with comments in the statusbar
# First we define a function to display a Formex and then wait for the user
# to click the Step button
def show(F,view='front',clearscr=True):
    if clearscr:
        clear()
    draw(F,view=view)
    pf.canvas.update()
    
# Here we go
message("Create a triangular pattern in the first octant")
f1 = Formex([[[0,0],[1,0]],[[1,0],[1,1]]]).replic2(8,8,1,1,0,1,1,-1) + Formex([[[1,0],[2,1]]]).replic2(7,7,1,1,0,1,1,-1)
show(f1)
#
message("Remove some of the bars")
f1 = f1.remove(Formex([[[2,0],[3,1]]]).replic(3,2,0))
show(f1)
#
message("Transform the octant into a circular sector")
f2 = f1.circulize1()
f1.setProp(1)
f2.setProp(2)
show(f1+f2)
clear()
#draw(f2)

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
pf.canvas.camera.setRotation(0,-45)
show(scallop(6,1,2,0),None,False)


howmany = ask("How many / Which domes do you want?",
              ['One','Sequence','Custom','None'])

n,f,c,r = [6,1,2.,0.]

if howmany == 'One':
   # The example from the pyformex homepage
   show(scallop(6,1,4,4),None)
        
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
        show(scallop(n,f,c,r),None)

elif howmany == 'Custom':
   # Customized version
   while True:
       res = askItems([['n',n],['f',f],['c',c],['r',r]])
       n = int(res['n'])
       f = int(res['f'])
       c = float(res['c'])
       r = float(res['r'])
       show(scallop(n,f,c,r),None)
       if not ack("Want to try another one?"):
           exit()


       
   
