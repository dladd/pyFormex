#!/usr/bin/env python
# $Id$
"""Element local coordinates and numbering.

The elements in this module are defined in a unit cell of their
local coordinates. See finite_elements.py for a variant using
natural coordinates familiar in Finite Element simulations.
"""

class Element(object):
    """Element base class: an empty element.

    Each element is defined by the following attributes:
    nodes: the natural coordinates of its nodes,
    edges: a list of edges, each defined by a couple of node numbers,
    faces: a list of faces, each defined by a list of minimum 3 node numbers,
    element: a list of all node numbers
    """
    
    nodes = [ [] ]
    edges = [ [] ]
    faces = [ [] ]
    element = []

    def nnodes(self):
        return len(self.nodes)
    def nedges(self):
        return len(self.edges)
    def nfaces(self):
        return len(self.faces)


class Tri3(Element):
    """A 3-node triangle"""
    nodes = [ [ 0.0, 0.0, 0.0 ],
              [ 1.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              ]

    edges = [ [0,1], [1,2], [2,0] ]

    faces = [ [0,1,2], ]

    element = faces[0]


class Quad4(Element):
    """A 4-node quadrilateral"""
    nodes = [ [  0.0,  0.0, 0.0 ],
              [  0.0,  1.0, 0.0 ],
              [  1.0,  1.0, 0.0 ],
              [  1.0,  0.0, 0.0 ],
              ]

    edges = [ [0,1], [1,2], [2,3], [3,0] ]

    faces = [ [0,1,2,3], ]

    element = faces[0]
    

class Tet4(Element):
    """A 4-node tetraeder"""
    nodes = [ [ 0.0, 0.0, 0.0 ],
              [ 1.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 1.0 ],
              ]

    edges = [ [0,1], [1,2], [2,0], [0,3], [1,3], [2,3] ]

    faces = [ [0,2,1], [0,1,3], [1,2,3], [2,0,3] ]

    element = [ 0,1,2,3 ]


class Hex8(Element):
    """An 8-node heaxeder"""

    nodes = [ [ 0.0, 0.0, 0.0 ],
              [ 1.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 1.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 0.0, 1.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ],
              ]
    
    edges = [ [0,1], [2,3], [4,5], [6,7],
              [0,2], [1,3], [4,6], [5,7],
              [0,4], [1,5], [2,6], [3,7] ]
    
    faces = [ [0,2,3,1], [4,5,7,6],
              [0,1,5,4], [2,6,7,3],
              [0,4,6,2], [1,3,7,5] ]

    element = [ 7,6,4,5,3,2,0,1, ]
