#!/usr/bin/pyformex --gui
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

"""ImportDxf

Import geometric entities from DXF files into pyFormex 
"""

from formex import *
from plugins import curve


Lines = []
Arcs = []
count = 0

def Line(x0,y0,z0,x1,y1,z1):
    global count
    Lines.append(Formex([[[x0,y0,z0],[x1,y1,z1]]],count))
    count += 1


def Arc(x0,y0,z0,r,a0,a1):
    global count
    Arcs.append(curve.Arc(center=[x0,y0,z0],radius=r,angles=[a0,a1]).setProp(count))
    count += 1


def readDXF(filename):
    import utils
    sta,out = utils.runCommand('dxfparser %s 2>/dev/null' % filename)
    if sta==0:
        return out
    else:
        return ''

def convertDXF(text):
    print text
    exec(text)
    return { 'Line':Lines, 'Arc':Arcs }


def importDXF(filename):
    text = readDXF(filename)
    print text
    if text:
        return convertDXF(text)
    else:
        return {}
    

def assembleLinesArcs(Lines,Arcs,ndiv=8):
    return Formex.concatenate([ a.toFormex() for a in Arcs ] + Lines)



# sample script if executed

if __name__ == "draw":

    clear()
    filename = askFilename(filter="AutoCAD .dxf files (*.dxf)")
    if not filename:
        exit()

    res = importDXF(filename)

    F = assembleLinesArcs(Lines,Arcs,ndiv=24)
    draw(F)
    drawPropNumbers(F)
    zoomAll()
    exit()

    M = F.toMesh()
    M.setProp(M.partitionByConnection())
    clear()
    draw(M)


# End
