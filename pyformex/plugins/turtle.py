#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
import math
rad = math.pi/180.
def sind(arg):
    """Return the sin of an angle in degrees."""
    return math.sin(arg*rad)
def cosd(arg):
    """Return the sin of an angle in degrees."""
    return math.cos(arg*rad)

global pos,step,angle,list
pos = [0,0]
step = 1
angle = 0
list=[]
save=[]

def reset():
    global pos,step,angle,list
    pos = [0,0]
    step = 1
    angle = 0
    list=[]
    save=[]
    #print "reset",pos,list

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
    #print "fd",pos,len(list)

def mv(d=None):
    fd(d,False)
    #print "mv",pos,len(list)

def ro(a):
    global pos,step,angle,list
    angle += a
    #print "ro",pos,len(list)

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
    

if __name__ == "__main__":
    def test(txt):
        l = play(txt)
        print len(l)," lines"
        
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("fd();ro(90);fd();ro(90);fd();ro(90);fd()")
    test("rs();fd();ro(90);fd();ro(90);fd();ro(90);fd()")
