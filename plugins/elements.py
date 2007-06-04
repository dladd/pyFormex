#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Sat Mar 10 20:05:55 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Element local coordinates and numbering.

The elements in this module are defined in a unit cell of their
local coordinates. See finite_elements.py for a variant using
natural coordinates familiar in Finite Element simulations.
"""
import math

golden_ratio = 0.5 * (1.0 + math.sqrt(5))


class Element(object):
    """Element base class: an empty element.

    Each element is defined by the following attributes:
    vertices: the natural coordinates of its vertices,
    edges: a list of edges, each defined by a couple of node numbers,
    faces: a list of faces, each defined by a list of minimum 3 node numbers,
    element: a list of all node numbers
    """
    
    vertices = []
    edges = []
    faces = []
    element = []

    def nvertices(self):
        return len(self.vertices)
    def nedges(self):
        return len(self.edges)
    def nfaces(self):
        return len(self.faces)


    # For compatibility with old version:
    nodes = vertices
    nnodes = nvertices


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
                 (  0.0,  1.0, 0.0 ),
                 (  1.0,  1.0, 0.0 ),
                 (  1.0,  0.0, 0.0 ),
                 ]

    edges = [ (0,1), (1,2), (2,3), (3,0) ]

    faces = [ (0,1,2,3), ]

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


class Hex8(Element):
    """An 8-node hexahedron"""

    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 1.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ( 1.0, 0.0, 1.0 ),
                 ( 0.0, 1.0, 1.0 ),
                 ( 1.0, 1.0, 1.0 ),
                 ]
    
    edges = [ (0,1), (2,3), (4,5), (6,7),
              (0,2), (1,3), (4,6), (5,7),
              (0,4), (1,5), (2,6), (3,7) ]
    
    faces = [ (0,2,3,1), (4,5,7,6),
              (0,1,5,4), (2,6,7,3),
              (0,4,6,2), (1,3,7,5) ]

    element = [ 7,6,4,5,3,2,0,1, ]


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


if __name__ == "__main__":

    print Icosa.vertices
