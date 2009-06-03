#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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
"""Element local coordinates and numbering.

This modules allows for a consistent local numbering scheme throughout
pyFormex. When interfacing with other programs, one should be aware
that conversions may be necessary. Conversions to/from external programs
should be done by the interface modules.
"""

import math
golden_ratio = 0.5 * (1.0 + math.sqrt(5))

import olist


class Element(object):
    """Element base class: an empty element.

    All derived classes should have a capitalized name: starting with
    an uppercase character and further only lower case and digits.

    Each element is defined by the following attributes:
    vertices: the natural coordinates of its vertices,
    edges: a list of edges, each defined by a couple of node numbers,
    faces: a list of faces, each defined by a list of minimum 3 node numbers,
    element: a list of all node numbers

    The vertices of the elements are defined in a unit space [0,1] in each
    axis direction. 

    The elements guarantee a fixed local numbering scheme of the vertices.
    One should however not rely on a specific numbering scheme of edges, faces
    or elements.
    For solid elements, it is guaranteed that the vertices of all faces are
    numbered in a consecutive order spinning positively around the outward
    normal on the face.
    """
    
    vertices = []
    edges = []
    faces = []
    element = []
    conversion = {}

    def nvertices(self):
        return len(self.vertices)
    def nedges(self):
        return len(self.edges)
    def nfaces(self):
        return len(self.faces)

    def getFaces():
        return olist.collectOnLength(self.faces)




class Tri3(Element):
    """A 3-node triangle"""
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,0) ]

    faces = [ (0,1,2), ]

    element = faces[0]


class Quad4(Element):
    """A 4-node quadrilateral"""
    vertices = [ (  0.0,  0.0, 0.0 ),
                 (  1.0,  0.0, 0.0 ),
                 (  1.0,  1.0, 0.0 ),
                 (  0.0,  1.0, 0.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,3), (3,0) ]

    faces = [ (0,1,2,3), ]

    element = faces[0]

    conversion = {
        'Tri3' : [ (0,1,2), (2,3,0) ],
        }


class Quad9(Element):
    """A 4-node quadrilateral"""
    vertices = [ (  0.0,  0.0, 0.0 ),
                 (  1.0,  0.0, 0.0 ),
                 (  1.0,  1.0, 0.0 ),
                 (  0.0,  1.0, 0.0 ),
                 (  0.5,  0.0, 0.0 ),
                 (  1.0,  0.5, 0.0 ),
                 (  0.5,  1.0, 0.0 ),
                 (  0.0,  0.5, 0.0 ),
                 (  0.5,  0.5, 0.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,3), (3,0) ]

    faces = [ (0,1,2,3,4,5,6,7,8), ]

    element = faces[0]


class Tet4(Element):
    """A 4-node tetrahedron"""
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) ]

    faces = [ (0,2,1), (0,1,3), (1,2,3), (2,0,3) ]

    element = [ 0,1,2,3 ]


class Wedge6(Element):
    """A 6-node wedge element"""
    vertices = [ ( 0.0, 0.0, 1.0 ),
                 ( 1.0, 0.0, 1.0 ),
                 ( 0.0, 1.0, 1.0 ),
                 ( 0.0, 0.0,-1.0 ),
                 ( 1.0, 0.0,-1.0 ),
                 ( 0.0, 1.0,-1.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,0), (0,3), (1,4), (2,5), (3,4), (4,5), (5,3) ]

    faces = [ (0,1,2), (3,5,4), (0,3,4,1), (1,4,5,2), (2,5,3,0) ]

    element = [ 0,1,2,3,4,5 ]


class Hex8(Element):
    """An 8-node hexahedron"""

    vertices = [ ( 0.0, 0.0, 0.0 ),  
                 ( 1.0, 0.0, 0.0 ),
                 ( 1.0, 1.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ( 1.0, 0.0, 1.0 ),
                 ( 1.0, 1.0, 1.0 ),
                 ( 0.0, 1.0, 1.0 ),
                 ]
    
    edges = [ (0,1), (1,2), (2,3), (3,0),
              (4,5), (5,6), (6,7), (7,4),
              (0,4), (1,5), (2,6), (3,7) ]
    
    faces = [ (0,4,7,3), (1,2,6,5),
              (0,1,5,4), (3,7,6,2),
              (0,3,2,1), (4,5,6,7) ]

    element = [ 0,1,2,3,4,5,6,7 ]


class Icosa(Element):
    """An icosahedron: a regular polyhedron with 20 triangular surfaces.

    nfaces = 20, nedges = 30, nvertices = 12
    """
    phi = golden_ratio
    
    vertices = [ ( 0.0, 1.0, phi ),
                 ( 0.0,-1.0, phi ),
                 ( 0.0, 1.0,-phi ),
                 ( 0.0,-1.0,-phi ),
                 ( 1.0, phi, 0.0 ),
                 (-1.0, phi, 0.0 ),
                 ( 1.0,-phi, 0.0 ),
                 (-1.0,-phi, 0.0 ),
                 ( phi, 0.0, 1.0 ),
                 ( phi, 0.0,-1.0 ),
                 (-phi, 0.0, 1.0 ),
                 (-phi, 0.0,-1.0 ),
                 ]

    edges = [ (0,1),  (0,8), (1,8), (0,10),(1,10),
              (2,3),  (2,9), (3,9), (2,11),(3,11),
              (4,5),  (4,0), (5,0), (4,2), (5,2),
              (6,7),  (6,1), (7,1), (6,3), (7,3),
              (8,9),  (8,4), (9,4), (8,6), (9,6),
              (10,11),(10,5),(11,5),(10,7),(11,7),
              ]
    
    faces = [ (0,1,8),  (1,0,10),
              (2,3,11), (3,2,9),
              (4,5,0),  (5,4,2),
              (6,7,3),  (7,6,1),
              (8,9,4),  (9,8,6),
              (10,11,7),(11,10,5),
              
              (0,8,4),  (1,6,8),
              (0,5,10), (1,10,7),
              (2,11,5), (3,7,11),
              (2,4,9),  (3,9,6),
              ]

    element = range(12)




if __name__ == "__main__":

    print Icosa.vertices
