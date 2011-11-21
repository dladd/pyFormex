#!/usr/bin/pyformex --gui
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

"""Script Template

This is a template file to show the general layout of a pyFormex script.
In the current version, a pyFormex script should obey the following rules:

- file name extension is '.py'
- first (comment) line contains 'pyformex'

The script starts by preference with a docstring (like this),
composed of a short first line, then a blank line and
one or more lines explaining the intention of the script.
"""
clear()
wireframe()

from plugins import curve, dxf_in

        
## Lines = []
## Arcs = []

## def Line(coords):
##     Lines.append(coords)
            
## def Arc(c,r,a):
##     Arcs.append(curve.Arc(center=c,radius=r,angles=a))

## def readDXF(filename):
##     import utils
##     sta,out = utils.runCommand('dxf_reader %s' % filename)
##     if sta==0:
##         print out
##         exec(out)


def assembleLinesArcs(Lines,Arcs,ndiv=8):
    Arcs = [ a.toFormex() for a in Arcs ] +  [ Lines ]
    return Formex.concatenate(Arcs) 


clear()
filename = askFilename(filter="AutoCAD .dxf files (*.dxf)")
if not filename:
    exit()

model = dxf_in.importDXF(filename)
F = assembleLinesArcs(model['Line'],model['Arc'],ndiv=24)
draw(F)
zoomAll()

M = F.toMesh()
M.setProp(M.partitionByConnection())
clear()
draw(M)


# End
