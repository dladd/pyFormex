#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Turtle graphics for pyFormex

This module was mainly aimed at the drawing of Lindenmayer products
(see :mod:`plugins.lima` and the Lima example).
"""
import math
rad = math.pi/180.
def sind(arg):
    """Return the sin of an angle in degrees."""
    return math.sin(arg*rad)
def cosd(arg):
    """Return the sin of an angle in degrees."""
    return math.cos(arg*rad)


def reset():
    """Reset the turtle graphics engine to start conditions.

    """
    global pos,step,angle,list,save
    pos = [0.,0.]
    step = 1.
    angle = 0.
    list=[]
    save=[]


def push():
    global save
    save.append([pos,step,angle])

def pop():
    global pos,step,angle,list,save
    pos,step,angle = save.pop(-1)

def fd(d=None,connect=True):
    global pos,step,angle,list
    if d:
        step = d
    p = [ pos[0] + step * cosd(angle), pos[1] + step * sind(angle) ]
    if connect:
        list.append([pos,p])
    pos = p


def mv(d=None):
    fd(d,False)


def ro(a):
    global pos,step,angle,list
    angle += a


def go(p):
    global pos,step,angle,list
    pos = p

def st(d):
    global pos,step,angle,list
    step = d

def an(a):
    global pos,step,angle,list
    angle = a

def play(scr,glob=None):
    import string
    for line in string.split(scr,";"):
        if line:
            if glob:
                glob.update(globals())
                eval(line,glob)
            else:
                eval(line)
    return list

reset()

if __name__ == "__main__":
    def test(txt):
        l = play(txt)
        print len(l)," line segments"
        print l
        
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("reset();fd();ro(90);fd();ro(90);fd();ro(90);fd()")
