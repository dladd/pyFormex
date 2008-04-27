#!/usr/bin/env pyformex --gui
# $Id$
"""Formex Structure

This script an image of how coordinates are structures in a Formex.
It was intended mainly for the manual.
"""
clear()

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
