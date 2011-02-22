#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""Definition of elements.

This modules allows for a consistent local numbering scheme of element
connectivities throughout pyFormex.
When interfacing with other programs, one should be aware
that conversions may be necessary. Conversions to/from external programs
should be done by the interface modules.
""" 
from numpy import array
from olist import collectOnLength
from odict import ODict


class Element(object):
    """Element base class: an empty element.

    All derived classes should have a capitalized name: starting with
    an uppercase character and further only lower case and digits.

    Each element is defined by the following attributes:

    - `vertices`: the natural coordinates of its vertices,
    - `edges`: a list of edges, each defined by a couple of node numbers,
    - `faces`: a list of faces, each defined by a list of minimum 3 node
      numbers,
    - `element`: a list of all node numbers
    - `drawfaces`: a list of faces to be drawn, if different from faces. This
      is an optional attribute. If defined, it will be used instead of the
      `faces` attribute to draw the element. This can e.g. be used to draw
      approximate representations for higher order elements for which there
      is no correct drawing function.

    The vertices of the elements are defined in a unit space [0,1] in each
    axis direction. 

    The elements guarantee a fixed local numbering scheme of the vertices.
    One should however not rely on a specific numbering scheme of edges, faces
    or elements.
    For solid elements, it is guaranteed that the vertices of all faces are
    numbered in a consecutive order spinning positively around the outward
    normal on the face.

    The list of available element types can be found by:
    
    >>> printElementTypes()
    Available Element Types:
      0-dimensional elements: ['point']
      1-dimensional elements: ['line2']
      2-dimensional elements: ['tri3', 'tri6', 'quad4', 'quad8', 'quad9']
      3-dimensional elements: ['tet4', 'tet10', 'wedge6', 'hex8', 'hex20', 'icosa']
    
    """
    proposed_changes = """..

    
Proposed changes in the Element class
=====================================

- nnodes = nvertices

- nodal coordinates are specified as follows:

  - in symmetry directions: between -1 and +1, centered on 0
  - in non-symmetry directions: between 0 and +1, aligned on 0

- getCoords() : return coords as is
- getAlignedCoords(): return coords between 0 ad 1 and aligned on 0 in all
  directions

- Make the elements into Element class instances instead of subclasses?


"""

    collection = ODict() # the full collection with all defined elements
    
    
    def __init__(self,name,doc,ndim,vertices,edges,faces,element=[],conversions={},drawedges=None,drawfaces=None,facetype=None,edgetype=None):
        self.doc = doc
        self.ndim = ndim
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.edgetype = edgetype
        self.facetype = facetype
        if not element:
            element = range(self.nplex())
        self.element = element
        self.conversions = conversions
        if drawedges:
            self.drawedges = drawedges
        if drawfaces:
            self.drawfaces = drawfaces
        # add to the collection
        name = name.lower()
        Element.collection[name] = self


    def nplex(self):
        return len(self.vertices)
    
    nvertices = nplex
    nnodes = nplex
    
    def nedges(self):
        return len(self.edges)
    
    def nfaces(self):
        return len(self.faces)

    def getFaces(self):
        return collectOnLength(self.faces)

    def toFormex(self):
        x = array(self.vertices)
        e = array(self.element)
        return Formex(x[e])

    def toMesh(self):
        from plugins import Mesh
        x = array(self.vertices)
        e = array(self.element)
        return Mesh(x,e)

    def name(self):
        for k,v in Element.collection.items():
            if v == self:
                return k
        return 'unregistered_element'

    def __str__(self):
        return self.name()

    def report(self):
        print "Element %s: ndim=%s, nplex=%s, nedges=%s, nfaces=%s" % (self.name(),self.ndim,self.nplex(),self.nedges(),self.nfaces())
  

    @classmethod
    def listAll(clas):
        return Element.collection.keys()
            
    @classmethod
    def printAll(clas):
        for k,v in Element.collection.items():
            print "Element %s: ndim=%s, nplex=%s, nedges=%s, nfaces=%s" % (k,v.ndim,v.nplex(),v.nedges(),v.nfaces())
        

# BV
# Should we use Coords objects for the vertices and arrays for the rest?

#####################################################
# Define the collection of default pyFormex elements 


## NoElement = Element(
##     'noelem',"No defined element type",
##     ndim = 0,
##     vertices = [],
##     edges = [],
##     faces = [],
##     )


Point = Element(
    'point',"A single point",
    ndim = 0,
    vertices = [ ( 0.0, 0.0, 0.0 ) ],
    edges = [ ],
    faces = [ ],
    )


Line2 = Element(
    'line2',"A 2-node line segment",
    ndim = 1,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ],
    edges = [ (0,1) ],
    faces = [ ],
    )


Line3 = Element(
    'line3',"A 3-node quadratic line segment",
    ndim = 1,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 0.5, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ],
    edges = [ (0,1,2) ], 
    faces = [ ],
    )


Tri3 = Element(
    'tri3',"A 3-node triangle",
    ndim = 2,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ],
    edges = [ (0,1), (1,2), (2,0) ], edgetype='line2', 
    faces = [ (0,1,2), ],
    )


Tri6 = Element(
    'tri6',"A 6-node triangle",
    ndim = 2,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.5, 0.0, 0.0 ),
                 ( 0.5, 0.5, 0.0 ),
                 ( 0.0, 0.5, 0.0 ),
                 ],    
    edges = [ (0,3,1), (1,4,2), (2,5,0) ], edgetype = 'line3', 
    faces = [ (0,1,2,3,4,5), ],
    drawfaces = [ (0,3,5),(3,1,4),(4,2,5),(3,4,5) ]
    )


Quad4 = Element(
    'quad4',"A 4-node quadrilateral",
    ndim = 2,
    vertices = [ (  0.0,  0.0, 0.0 ),
                 (  1.0,  0.0, 0.0 ),
                 (  1.0,  1.0, 0.0 ),
                 (  0.0,  1.0, 0.0 ),
                 ],
    edges = [ (0,1), (1,2), (2,3), (3,0) ], edgetype='line2', 
    faces = [ (0,1,2,3), ],
    )


Quad8 = Element(
    'quad8',"A 8-node quadrilateral",
    ndim = 2,
    vertices = Quad4.vertices + [ (  0.5,  0.0, 0.0 ),
                                  (  1.0,  0.5, 0.0 ),
                                  (  0.5,  1.0, 0.0 ),
                                  (  0.0,  0.5, 0.0 ),
                                  ],
    edges = [ (0,4,1), (1,5,2), (2,6,3), (3,7,0), ], edgetype = 'line3', 
    faces = [ (0,1,2,3,4,5,6,7), ],
#    drawedges = [ (0,4), (4,1), (1,5), (5,2), (2,6), (6,3), (3,7), (7,0) ],
    drawfaces = [(0,4,7), (1,5,4), (2,6,5), (3,7,6), (4,5,6), (4,6,7) ],
)

Quad9 = Element(
    'quad9',"A 9-node quadrilateral",
    ndim = 2,
    vertices = Quad8.vertices + [ (  0.5,  0.5, 0.0 ), ],
    edges = Quad8.edges, edgetype=Quad8.edgetype, 
    faces = [ (0,1,2,3,4,5,6,7,8), ],
    drawfaces = [(0,4,8),(4,1,8),(1,5,8),(5,2,8),(2,6,8),(6,3,8),(3,7,8),(7,0,8) ],
)

######### 3D ###################

Tet4 = Element(
    'tet4',"A 4-node tetrahedron",
    ndim = 3,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ],
    edges = [ (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) ], edgetype='line2', 
    faces = [ (0,2,1), (0,1,3), (1,2,3), (2,0,3) ],
    )


Tet10 = Element(
    'tet10',"A 10-node tetrahedron",
    ndim = 3,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ( 0.5, 0.0, 0.0 ),
                 ( 0.0, 0.5, 0.0 ),
                 ( 0.0, 0.0, 0.5 ),
                 ( 0.5, 0.5, 0.0 ),
                 ( 0.0, 0.5, 0.5 ),
                 ( 0.5, 0.0, 0.5 ),
                 #( 0.5, 0.5, 0.5 ),#GDS this is not needed!
                 ],
# BV: This needs further specification!
    edges = Tet4.edges,
    faces = Tet4.faces,
    )


Wedge6 = Element(
    'wedge6',"A 6-node wedge element",
    ndim = 3,
    vertices = [ ( 0.0, 0.0, 1.0 ),
                 ( 1.0, 0.0, 1.0 ),
                 ( 0.0, 1.0, 1.0 ),
                 ( 0.0, 0.0,-1.0 ),
                 ( 1.0, 0.0,-1.0 ),
                 ( 0.0, 1.0,-1.0 ),
                 ],
    edges = [ (0,1), (1,2), (2,0), (0,3), (1,4), (2,5), (3,4), (4,5), (5,3) ],
    faces = [ (0,1,2), (3,5,4), (0,3,4,1), (1,4,5,2), (2,5,3,0) ],
    )


Hex8 = Element(
    'hex8',"An 8-node hexahedron",
    ndim = 3,
    vertices = [ ( 0.0, 0.0, 0.0 ),  
                 ( 1.0, 0.0, 0.0 ),
                 ( 1.0, 1.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ( 1.0, 0.0, 1.0 ),
                 ( 1.0, 1.0, 1.0 ),
                 ( 0.0, 1.0, 1.0 ),
                 ],
    edges = [ (0,1), (1,2), (2,3), (3,0),
              (4,5), (5,6), (6,7), (7,4),
              (0,4), (1,5), (2,6), (3,7) ],
    faces = [ (0,4,7,3), (1,2,6,5),
              (0,1,5,4), (3,7,6,2),
              (0,3,2,1), (4,5,6,7) ],
    )


Hex20 = Element(
    'hex20',"An 20-node hexahedron",
    ndim = 3,
    vertices = Hex8.vertices + [
        (  0.5,  0.0, 0.0 ),
        (  1.0,  0.5, 0.0 ),
        (  0.5,  1.0, 0.0 ),
        (  0.0,  0.5, 0.0 ),
        (  0.5,  0.0, 1.0 ),
        (  1.0,  0.5, 1.0 ),
        (  0.5,  1.0, 1.0 ),
        (  0.0,  0.5, 1.0 ),
        (  0.0,  0.0, 0.5 ),
        (  1.0,  0.0, 0.5 ),
        (  1.0,  1.0, 0.5 ),
        (  0.0,  1.0, 0.5 )
        ],
    edges = [ (0,8,1), (1,9,2), (2,10,3),(3,11,0),
              (4,12,5),(5,13,6),(6,14,7),(7,15,4),
              (0,16,4),(1,17,5),(2,18,6),(3,19,7) ],
    faces = [ (0,4,7,3,16,15,19,11), (1,2,6,5,9,18,13,17),
              (0,1,5,4,8,17,12,16), (3,7,6,2,19,14,18,10),
              (0,3,2,1,11,10,9, 8), (4,5,6,7,12,13,14,15) ],
)
# We can not set this at __init__ time
Hex20.drawfaces = array(Hex20.faces)[:, Quad8.drawfaces].reshape(-1, 3)


########## element type conversions ##################################


Line3.conversions = {
    'line2'  : [ ('s', [ (0,2) ]), ],
    'line2-2' : [ ('s', [ (0,1), (1,2) ]), ],
    }
Tri3.conversions =  {
    'tri3-4' : [ ('v', 'tri6'), ],
    'tri6'   : [ ('m', [ (0,1), (1,2), (2,0) ]), ],
    'quad4'  : [ ('v', 'tri6'), ],
    }
Tri6.conversions = {
    'tri3'   : [ ('s', [ (0,1,2) ]), ],
    'tri3-4' : [ ('s', [ (0,3,5),(3,1,4),(4,2,5),(3,4,5) ]), ],
    'quad4'  : [ ('m', [ (0,1,2), ]),
                 ('s', [ (0,3,6,5),(1,4,6,3),(2,5,6,4) ]),
                 ],
    }
Quad4.conversions = {
    'tri3'   : 'tri3-u',
    'tri3-r' : [ ('r', ['tri3-u','tri3-d']), ],
    'tri3-u' : [ ('s', [ (0,1,2), (2,3,0) ]), ],
    'tri3-d' : [ ('s', [ (0,1,3), (2,3,1) ]), ],
    'tri3-x' : [ ('m', [ (0,1,2,3) ]),
                 ('s', [ (0,1,4),(1,2,4),(2,3,4),(3,0,4) ]),
                 ],
    'quad8'  : [ ('m', [ (0,1), (1,2), (2,3), (3,0) ]), ],
    'quad4-4': [ ('v', 'quad9'), ],
    'quad9'  : [ ('v', 'quad8'), ],
    }
Quad8.conversions = {
    'tri3'   : [ ('v', 'quad9'), ],
    'tri3-v' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(5,6,4),(7,4,6) ]), ],
    'tri3-h' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(4,5,7),(6,7,5) ]), ],
    'quad4'  : [ ('s', [ (0,1,2,3) ]), ],
    'quad4-4': [ ('v', 'quad9'), ],
    'quad9'  : [ ('m', [ (4,5,6,7) ]), ],
    }
Quad9.conversions = {
    'quad8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
    'quad4'  : [ ('v', 'quad8'), ],
    'quad4-4': [ ('s', [ (0,4,8,7),(4,1,5,8),(7,8,6,3),(8,5,2,6) ]), ],
    'tri3'   : 'tri3-d',
    'tri3-d' : [ ('s', [ (0,4,7),(4,1,5),(5,2,6),(6,3,7),
                         (7,4,8),(4,5,8),(5,6,8),(6,7,8) ]), ],
    'tri3-x' : [ ('s', [ (0,4,8),(4,1,8),(1,5,8),(5,2,8),
                         (2,6,8),(6,3,8),(3,7,8),(7,0,8) ]), ],
    }
Tet4.conversions = {
    'tet10' : [ ('m', [ (0,1), (0,2), (0,3), (1,2), (2, 3), (1, 3)]), ],
    }
Tet10.conversions = {
    'tet4' :  [ ('s', [ (0,1,2,3,) ]), ],
    #'hex8'#TODO
    }
Wedge6.conversions = {
    'tet4'  : [ ('s', [ (0,1,2,3),(1,2,3,4),(2,3,4,5) ]), ],
    }
Hex8.conversions = {
    'wedge6': [ ('s', [ (0,1,2,4,5,6),(2,3,0,6,7,4) ]), ],
    'tet4'  : [ ('s', [ (0,1,2,5),(2,3,0,7),(5,7,6,2),(7,5,4,0),(0,5,2,7) ]), ],
    'hex20' : [ ('m', [ (0,1), (1,2), (2,3), (3,0),
                        (4,5), (5,6), (6,7), (7,4),
                        (0,4), (1,5), (2,6), (3,7), ]), ],
    }
Hex20.conversions = {
    'hex8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
    'tet4'  : [ ('v', 'hex8'), ],
    }



##########################################################  
# This element added just for fun, no practical importance

from arraytools import golden_ratio as phi

Icosa = Element(
    'icosa',
    """An icosahedron: a regular polyhedron with 20 triangular surfaces.,

    nfaces = 20, nedges = 30, nvertices = 12
    """,
    ndim = 3,
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
                 ],
    edges = [ (0,1),  (0,8), (1,8), (0,10),(1,10),
              (2,3),  (2,9), (3,9), (2,11),(3,11),
              (4,5),  (4,0), (5,0), (4,2), (5,2),
              (6,7),  (6,1), (7,1), (6,3), (7,3),
              (8,9),  (8,4), (9,4), (8,6), (9,6),
              (10,11),(10,5),(11,5),(10,7),(11,7),
              ],
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
              ],
    )


# list of default element type per plexitude
_default_eltype = {
    1 : Point,
    2 : Line2,
    3 : Tri3,
    4 : Quad4,
    6 : Wedge6,
    8 : Hex8,
    }

_default_edgetype = {
    2 : 'line2',
    3 : 'line3',
    }

_default_facetype = {
    3 : 'tri3',
    4 : 'quad4',
    6 : 'tri6',
    8 : 'quad8',
    9 : 'quad9',
    }



def elementType(name=None,nplex=-1):
    """Return the requested element type

    Parameters:

    - `name`: a string (case ignored) with the name of an element.
      If not specified, or the named element does not exist, the default
      element for the specified plexitude is returned.
    - `nplex`: plexitude of the element. If specified and no element name
      was given, the default element type for this plexitude is returned.

    Returns: a subclass of :class:`Element`

    Errors: if neither `name` nor `nplex` can resolve into an element type,
      an error is raised.

    Example:
    
    >>> elementType('tri3').name()
    'tri3'
    >>> elementType(nplex=2).name()
    'line2'
    """
    ## >>> try:
    ## ...     elementType('tri3',4)
    ## ... except:
    ## ...     print "No such element"
    ## No such element


    if isinstance(name,Element):
        return name
    
    
    eltype = None
    try:
        eltype = Element.collection[name.lower()]
        #print "TEST %s (%s)" % (eltype,nplex)
        # TESTING WOULD BREAK SOME INVALID ELTYPE SETTINGS in mesh
        # MAYBE INTRODUCE None/INvalidElement for no valid eltype
        #if not (nplex >= 0 and nplex != eltype.nplex()):
        return eltype
    except:
        pass
    if eltype is None:
        try:
            return _default_eltype[nplex]
        except:
            pass

    return None

    #return NoElement
    
    #raise ValueError("There is no element with name '%s' and plexitude '%s'" % (str(name),str(nplex)))


def elementTypes(ndim=None):
    """Return the names of available elements.

    If a value is specified for ndim, only the elements with the matching
    dimensionality are returned.
    """
    if ndim is None:
        return Element.collection.keys()
    else:
        return [ k for k,v in Element.collection.items() if v.ndim==ndim] 

def printElementTypes():
    """Print all available element types.

    Prints a list of the names of all availabale element types,
    grouped by their dimensionality.
    """
    print "Available Element Types:"        
    for ndim in range(4):
        print "  %s-dimensional elements: %s" % (ndim,elementTypes(ndim)        )



if __name__ == "__main__":
    printElementTypes()

# End
