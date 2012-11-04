# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Shapes

This example illustrates how pyFormex can be used in education.
I started this example when my daughter learned programming at school.
She had to create this image using Java(Beans).
It took days of wrestling with complex programming environments, compiling
multiple files, endless checking and debugging java declaraions and code.
I showed her how I could get to the same result with just a few lines of
pyFormex code.

Later I added some nice utilities to make it worthwile as a programming example.
It would be nice a nice exercise to add a GUI to create and position the shapes.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['illustration']
_techniques = ['animation']

from gui.draw import *
   

def circle(n=60):
    a1 = 360./n
    return Formex([[[cosd(i*a1),sind(i*a1),0.] for i in range(n)]])

def triangle():
    return Formex([[[0.,0.,0.],[1.,0.,0.],[0.5,0.5*sqrt(3.),0.]]])

def square():
    return Formex([[[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]])


class Shape(Geometry):
    def __init__(self,shape,size,position,color):
        self.shape = shape
        self.size = resize(size,(3))
        self.position = position
        self.color = color
        self.F = None
        self.A = None
        self.make()

    def make(self):
        self.F = globals()[self.shape]().scale(self.size).translate(self.position)

    def draw(self):
        self.A = draw(self.F,color=self.color)

    def hide(self):
        if self.A:
            undraw(self.A)
            self.A = None

    def redraw(self):
        A = self.A
        self.draw()
        undraw(A)

    def setSize(size):
        self.size = size
        self.make()

    def setPosition(pos):
        self.position = pos
        self.make()

    def setColor(color):
        self.hide()
        self.color = color
        self.draw()

    def move(self,direction,step):
        self.F = self.F.trl(direction,step)


def run():
    clear()
    flat()

    wall = Shape('square',[80.,60.],[10.,0.],'red')
    window = Shape('square',[10.,10.],[30.,30.],'white')
    roof = Shape('triangle',[100.,40.],[0.,60.],'green')
    sun = Shape('circle',10,[110.,80.],'yellow')

    delay(0)
    window.draw()
    wall.draw()
    roof.draw()
    sun.draw()
    zoomAll()

    # lower the sun
    n = 100
    delay(10./n)
    setDrawOptions({'bbox':None})
    sun.hide()
    for y in range(n):
        sun.move(0,sqrt(0.4*(n-y)/n))
        sun.move(1,-100./n)
        sun.redraw()

        
    sun.hide()
    
if __name__ == 'draw':
    run()
# End
