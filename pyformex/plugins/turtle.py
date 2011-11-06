#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
"""Turtle graphics for pyFormex

This module was mainly aimed at the drawing of Lindenmayer products
(see :mod:`plugins.lima` and the Lima example).

The idea is that a turtle can be moved in 2D from one position to another,
thereby creating a line between start and endpoint or not.

The current state of the turtle is defined by

- ``pos``: the position as a 2D coordinate pair (x,y),
- ``angle``: the moving direction as an angle (in degrees) with the x-axis,
- ``step``: the speed, as a discrete step size.

The start conditions are: ``pos=(0,0), step=1., angle=0.``

The followin example turtle script creates a unit square::

  fd();ro(90);fd();ro(90);fd();ro(90);fd()
"""

import math
deg = math.pi/180.

def sind(arg):
    """Return the sine of an angle in degrees."""
    return math.sin(arg*deg)

def cosd(arg):
    """Return the cosine of an angle in degrees."""
    return math.cos(arg*deg)


def reset():
    """Reset the turtle graphics engine to start conditions.

    This resets the turtle's state to the starting conditions
    ``pos=(0,0), step=1., angle=0.``, removes everything from the state save
    stack and empties the resulting path.
    """
    global pos,step,angle,list,save
    pos = [0.,0.]
    step = 1.
    angle = 0.
    list=[]
    save=[]


def push():
    """Save the current state of the turtle.

    The turtle state includes its position, step and angle.
    """
    global save
    save.append([pos,step,angle])

def pop():
    """Restore the turtle state to the last saved state.""" 
    global pos,step,angle,list,save
    pos,step,angle = save.pop(-1)


def fd(d=None,connect=True):
    """Move forward over a step `d`, with or without drawing.

    The direction is the current direction.
    If `d` is not given, the step size is the current step.

    By default, the new position is connected to the previous with a
    straight line segment.
    """
    global pos,step,angle,list
    if d:
        step = d
    p = [ pos[0] + step * cosd(angle), pos[1] + step * sind(angle) ]
    if connect:
        list.append([pos,p])
    pos = p


def mv(d=None):
    """Move over step `d` without drawing."""
    fd(d,False)


def ro(a):
    """Rotate over angle `a`. The new direction is incremented with `a`"""
    global pos,step,angle,list
    angle += a


def go(p):
    """Go to position `p` (without drawing).

    While the `mv` method performs a relative move, this is an absolute move.
    `p` is a tuple of (x,y) values.
    """
    global pos,step,angle,list
    pos = p

def st(d):
    """Set the step size."""
    global pos,step,angle,list
    step = d

def an(a):
    """Set the angle"""
    global pos,step,angle,list
    angle = a


def play(scr,glob=None):
    """Play all the commands in the script `scr`

    The script is a string of turtle commands, where each command
    is ended with a semicolon (';').

    If a dict `glob` is specified, it will be update with the turtle
    module's globals() after each turtle command.
    """
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
        print("%s line segments" % len(l))
        print(l)
        
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("reset();fd();ro(90);fd();ro(90);fd();ro(90);fd()")

# End
