#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Formex Structure

level = 'beginner'
topics = ['manual']
techniques = ['draw']

This script creates an image of how coordinates are structures in a Formex.
It was intended mainly for the manual.
"""
clear()

def tmbbox(a):
    return [[0.0,0.0,0.0],[1.0,1.0,1.0]]

marks.TextMark.bbox = tmbbox

def drawText3D(P,text,color=colors.black,font=None):
    """Draw a text at a 3D point."""
    M = marks.TextMark(P,text,color=color,font=font)
    GD.canvas.addActor(M)
    GD.canvas.update()
    return M


def drawAxis(len,dir,text):
    """Draw an axis of given length and direction annotated with text."""
    F = Formex(pattern('1')).scale(len).rotate(dir)
    #T = F[0][1].scale(1.1)
    draw(F,linewidth=2.0)
    drawText3D(F[0][1]+(2.,-0.5,0.),text,font='tr24')
    return F

def drawFrame(P):
    """Draw a dashed frame at position P."""
    d,e = (2,3) # dash length and step
    h = Formex(pattern('1')).scale(d)
    v = h.rotate(-90).replic(4,-e,1)
    h = h.replic(6,e,0)
    frame = (h + v).trl(P)
    draw(frame,linewidth=1.0,bbox=None)

drawAxis(30,0,'axis 2: coordinates')
drawAxis(30,-90,'axis 1: points')
F = drawAxis(50,30,'axis 0: elements').divide(8)

for i in range(1,5,2):
    drawFrame(F[i][1])

zoomAll()
