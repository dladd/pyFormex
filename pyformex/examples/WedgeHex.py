#!/usr/bin/env pyformex --gui
# $Id: Isopar.py 921 2009-02-24 09:51:50Z bverheg $
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""WedgeHex

level = 'advanced'
topics = ['FEA']
techniques = ['meshing'] 
"""

import simple
import elements
from plugins.mesh import *
from plugins.fe import *


_degen_hex8_wedge6 = [
    ([[0,1],[4,5]], [0,2,3,4,6,7]),
    ([[1,2],[5,6]], [0,1,3,4,5,7]),
    ([[2,3],[6,7]], [0,1,2,4,5,6]),
    ([[3,0],[7,4]], [0,1,2,4,5,6]),
    ([[0,1],[3,2]], [0,4,5,3,7,6]),
    ([[1,5],[2,6]], [0,4,5,3,7,6]),
    ([[5,4],[6,7]], [0,4,1,3,7,2]),
    ([[4,0],[7,3]], [0,5,1,3,6,2]),
    ([[0,3],[1,2]], [0,7,4,1,6,5]),
    ([[3,7],[2,6]], [0,3,4,1,2,5]),
    ([[7,4],[6,5]], [0,3,4,1,2,5]),
    ([[4,0],[5,1]], [0,3,7,1,2,6]),
    ]


def degenerate_hex8_wedge6(e):
    print e
    for conditions,selector in _degen_hex8_wedge6:
        cond = array(conditions)
        print cond
        print e[cond[:,0]]
        print e[cond[:,1]]
        if (e[cond[:,0]] == e[cond[:,1]]).all():
            print "MATCH"
            return e[selector]
    return None

def degenerate_hex8(m):
    """Test for a degenerate element that yields another non-degenerate."""
    faces = m.getFaces()
    deg = faces.testDegenerate().reshape(m.nelems(),6)
    faces = faces.reshape(m.nelems(),6,4)
    print faces
    degsum = deg.sum(axis=1)
    # the ones we can fix
    ok = degsum==3
    #
    okmesh = []
    wedge6 = []
    for i in where(ok)[0]:
        print "FIXIN %s" % i
        print "elems",m.elems[i]
        print "faces",faces[i]
        print "deg",deg[i]
        
        fixed = degenerate_hex8_wedge6(m.elems[i])
        if fixed is not None:
            wedge6.append(fixed)
            print "OK"


    if len(wedge6) > 0:
        w = array(wedge6)
        print w.shape
        print w
        print m.coords.shape
        okmesh.append(Mesh(m.coords,w,eltype='wedge6'))
    return okmesh
        

def splitDegenerate(self,autofix=True):
    """Split a Mesh in degenerate and non-degenerate elements.

    If autofix is True, the degenerate elements will be tested against
    known degeneration patterns, and the matching elements will be transformed
    to non-degenerate elements of a lower plexitude.

    The return value is a list of Meshes. The first holds the non-degenerate
    elements of the original Mesh. The last holds the remaining degenerate
    elements. The intermediate Meshes, if any, hold non-degenerate elements
    of a lower plexitude than the original.
    """
    deg = self.elems.testDegenerate()
    M0 = self.select(~deg)
    M1 = self.select(deg)

    print M1
    e = M1.elems[0]
    print e
    w6 = degenerate_hex8(M1)
    
    ML = [M0] + w6
    return ML

Mesh.splitDegenerate = splitDegenerate

clear()
smoothwire()

#create a 2D xy mesh
n = 4
G = simple.rectangle(1,1,1.,1.).replic2(n,n)
M = G.toMesh()
draw(M, color='red')
view('front')

#create a 3D axial-symmetric mesh by REVOLVING
parts=[]

R = M.revolve(n=8, angle=40)
draw(R,color='yellow')


sleep(5)
clear()
ML = R.splitDegenerate()
ML = [ Mi.setProp(i) for i,Mi in enumerate(ML) ]
draw(ML)
