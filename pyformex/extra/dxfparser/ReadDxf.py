#!/usr/bin/pyformex --gui
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
from __future__ import print_function
clear()
wireframe()
from pyformex.plugins.geomtools import anyPerpendicularVector as av

print(av([0.,0.,1.]))
print(av([0.,1.,1.]))
print(av([1.,1.,1.]))
print(av([1.,0.,1.]))
print(av([1.,0.,0.]))
print(av([0.,1.,0.]))
exit()



from plugins.curve import *
from plugins.dxf import *


class LineDrawing(object):
    """A collection of curves.

    """
    def __init__(self,data):
        self.all = data
        print(data)
        print([ type(i) for i in data ])
        print([ i.__class__ for i in data ])
        print(Arc,Line,PolyLine)
        self.lines = [ i for i in data if i.__class__ == Line ]
        self.polylines = [  i for i in data if type(i) is PolyLine ]
        self.arcs = [ i for i in data if type(i) is Arc ]
        self.report()


    def report(self):
        print("LineDrawing: %s Lines, %s PolyLines, %s Arcs" % (len(self.lines),len(self.polylines),len(self.arcs)))
 

    def assembleCurves(self):
        print(len(self.all))
        FL = [Formex([a.endPoints()]).setProp(i) for i,a in enumerate(self.all)]
        FL = Formex.concatenate(FL)
        print(FL.shape())
        print(FL)
        M = FL.toMesh()
        self.parts = M.partitionByConnection()
        return M.setProp(self.parts)


    def setProp(self,prop):
        for a,p in zip(self.all,prop):
            print(a,p)
            a.setProp(p)

        

if ack('Custom file?'):
    fn = askFilename(filter=utils.fileDescription('dxf'))
    if not fn:
        exit()
    model = importDXF(fn)
else:
    fn = os.path.join(pf.cfg['pyformexdir'],'data','P.dxftext')
    model = convertDXF(open(fn).read())

for a in model:
    print(a.prop)
draw(model)

D = LineDrawing(model)
D.assembleCurves()
print(D.parts)
#D.setProp(D.parts)
clear()
for a,c in zip(D.all,D.parts):
    draw(a,color=c)




# End
