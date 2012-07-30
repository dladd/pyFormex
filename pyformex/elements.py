# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
"""Definition of elements.

This modules allows for a consistent local numbering scheme of element
connectivities throughout pyFormex.
When interfacing with other programs, one should be aware
that conversions may be necessary. Conversions to/from external programs
should be done by the interface modules.
"""
import pyformex
from coords import Coords
from connectivity import Connectivity
from numpy import array,arange
from odict import ODict

class TestP(object):
    c = 4

    def __getstate__(self):
        state = {'d':"hallo"}
        print "PICKLING",state
        return state

    def __setstate__(self,state):
        print "UNPICKLING",state
        self.__dict__.update(state)

def _sanitize(ent):
    # input is Connectivity or (eltype,table)
    # output is Connectivity
    if isinstance(ent,Connectivity):
        if hasattr(ent,'eltype'):
            return ent
        else:
            raise ValueError,"Conectivity should have an element type"
    else:
        return Connectivity(ent[1],eltype=ent[0])


class ElementType(object):
    """Base class for element type classes.

    Element type data are stored in a class derived from ElementType.
    The derived element type classes contain only static data. No instances
    of these classes should be created. The base class defines the access
    methods, which are all class methods.

    Derived classes should be created by calling the function
    :func:`createElementType.

    Each element is defined by the following attributes:

    - `name`: a string. It is capitalized before use, thus all ElementType
      subclasses have a name starting with an uppercase letter. Usually the
      name has a numeric last part, equal to the plexitude of the element. 

    - `vertices`: the natural coordinates of its vertices,
    
    - `edges`: a list of edges, each defined by 2 or 3 node numbers,
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
      1-dimensional elements: ['line2', 'line3']
      2-dimensional elements: ['tri3', 'tri6', 'quad4', 'quad6', 'quad8', 'quad9']
      3-dimensional elements: ['tet4', 'tet10', 'tet14', 'tet15', 'wedge6', 'hex8', 'hex16', 'hex20', 'icosa']

    Optional attributes:

    - `conversions`: Defines possible strategies for conversion of the element
      to other element types. It is a dictionary with the target element name
      as key, and a list of actions as value. Each action in the list consists
      of a tuple ( action, data ), where action is one of the action identifier
      characters defined below, and data are the data needed for this action.

    Conversion actions:

    'm': add new nodes to the element by taking the mean values of existing
         nodes. data is a list of tuples containing the nodes numbers whose
         coorrdinates have to be averaged.
    's': select nodes from the existing ones. data is a list of the node numbers
         to retain in the new element. This can be used to reduce the plexitude
         but also just to reorder the existing nodes.
    'v': perform a conversion via an intermediate type. data is the name of the
         intermediate element type. The current element will first be converted
         to the intermediate type, and then conversion from that type to the
         target will be attempted. 
    'r': randomly choose one of the possible conversions. data is a list of
         element names. This can e.g. be used to select randomly between
         different but equivalent conversion paths.

    """
    
   
## Proposed changes in the Element class
## =====================================

## - nodal coordinates are specified as follows:

##   - in symmetry directions: between -1 and +1, centered on 0
##   - in non-symmetry directions: between 0 and +1, aligned on 0

## - getCoords() : return coords as is
## - getAlignedCoords(): return coords between 0 ad 1 and aligned on 0 in all
##   directions

    name = None
    doc = "NO ELEMENT TYPE"
    ndim = 0
    vertices = []
    edges = []
    faces = []

    @classmethod
    def nplex(self):
        """Return the plexitude of the element"""
        return self.vertices.shape[0]
    
    nvertices = nplex
    nnodes = nplex

                                      
    @classmethod
    def nedges(self):
        return self.edges.nelems()
    
    @classmethod
    def nfaces(self):
        return self.faces.nelems()


    @classmethod
    def getPoints(self):
        return self.getEntities(0)

    @classmethod
    def getEdges(self):
        return self.getEntities(1)

    @classmethod
    def getFaces(self):
        return self.getEntities(2)

    @classmethod
    def getCells(self):
        return self.getEntities(3)
    
    @classmethod
    def getElement(self):
        return self.getEntities(self.ndim)


    @classmethod
    def getEntities(self,level,reduce=False):
        """Return the type and connectivity table of some element entities.

        The full list of entities with increasing dimensionality  0,1,2,3 is::

            ['points', 'edges', 'faces', 'cells' ]

        If level is negative, the dimensionality returned is relative
        to the highest dimensionality (.i.e., that of the element).
        If it is positive, it is taken absolute.

        Thus, for a 3D element type, getEntities(-1) returns the faces,
        while for a 2D element type, it returns the edges.
        For both types however, getLowerEntities(+1) returns the edges.

        The return value is a dict where the keys are element types
        and the values are connectivity tables. 
        If reduce == False: there will be only one connectivity table
        and it may include degenerate elements.
        If reduce == True, an attempt is made to reduce the degenerate
        elements. The returned dict may then have multiple entries.
        
        If the requested entity level is outside the range 0..ndim,
        the return value is None.
        """
        if level < 0:
            level = self.ndim + level

        if level < 0 or level > self.ndim:
            return Connectivity()
        
        if level == 0:
            return Connectivity(arange(self.nplex()).reshape((-1,1)),eltype='point')

        elif level == self.ndim:
            return Connectivity(arange(self.nplex()).reshape((1,-1)),eltype=self)

        elif level == 1:
            return self.edges

        elif level == 2:
            return self.faces
 

    @classmethod
    def getDrawEdges(self,quadratic=False):
        if quadratic and hasattr(self,'drawedges2'):
            return self.drawedges2
        if not hasattr(self,'drawedges'):
            self.drawedges = self.getEdges().reduceDegenerate()
        return self.drawedges


    @classmethod
    def getDrawFaces(self,quadratic=False):
        """Returns the local connectivity for drawing the element's faces"""
        if quadratic and hasattr(self,'drawfaces2'):
            return self.drawfaces2
        if not hasattr(self,'drawfaces'):
            self.drawfaces = self.getFaces().reduceDegenerate()
        return self.drawfaces

    @classmethod
    def toMesh(self):
        """Convert the element type to a Mesh.

        Returns a Mesh with a single element of natural size.
        """
        from mesh import Mesh
        x = self.vertices
        e = self.getElement()
        return Mesh(x,e,eltype=e.eltype)
        

    @classmethod
    def toFormex(self):
        """Convert the element type to a Formex.

        Returns a Formex with a single element of natural size.
        """
        return self.toMesh().toFormex()


    @classmethod
    def name(self):
        """Return the lowercase name of the element.

        For compatibility, name() returns the lower case version of the
        ElementType's name. To get the real name, use the attribute
        `__name__` or format the ElementType as a string.
        """
        return self.__name__.lower()


    @classmethod
    def __str__(self):
        return self.__name__

    @classmethod
    def __repr__(self):
        return "elementType(%s)" % self.__name__

    @classmethod
    def report(self):
        return "ElementType %s: ndim=%s, nplex=%s, nedges=%s, nfaces=%s" % (self.__name__,self.ndim,self.nplex(),self.nedges(),self.nfaces())


# all registered element types:
_registered_element_types = {}


def createElementType(name,doc,ndim,vertices,edges=('',[]),faces=('',[]),**kargs):
    name = name.capitalize()
    if name in _registered_element_types:
        raise ValueError,"Element type %s already exists" % name

    #print "\n CREATING ELEMENT TYPE %s\n" % name
    
    D = dict(
        __doc__ = doc,
        ndim = ndim,
        vertices = Coords(vertices),
        edges = _sanitize(edges),
        faces = _sanitize(faces),
        )

    for a in [ 'drawedges', 'drawedges2', 'drawfaces', 'drawfaces2']:
        if a in kargs:
            D[a] = [ _sanitize(e) for e in kargs[a] ]
            del kargs[a]

    # other args are added as-is        
    D.update(kargs)
    #print "Final class dict:",D

        ## # add the element to the collection
        ## if self._name in Element.collection:
        ##     raise ValueError,"Can not create duplicate element names"
        ## Element.collection[self._name] = self

    C = type(name,(ElementType,),D)
    _registered_element_types[name] = C
    return C

#####################################################
# Define the collection of default pyFormex elements 

Point = createElementType(
    'point',"A single point",
    ndim = 0,
    vertices = [ ( 0.0, 0.0, 0.0 ) ],
    )

Line2 = createElementType(
    'line2',"A 2-node line segment",
    ndim = 1,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ],
    )

Line3 = createElementType(
    'line3',"A 3-node quadratic line segment",
    ndim = 1,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 0.5, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ],
    )

######### 2D ###################

Tri3 = createElementType(
    'tri3',"A 3-node triangle",
    ndim = 2,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ],
    edges = ('line2', [ (0,1), (1,2), (2,0) ])
    )

Tri6 = createElementType(
    'tri6',"A 6-node triangle",
    ndim = 2,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.5, 0.0, 0.0 ),
                 ( 0.5, 0.5, 0.0 ),
                 ( 0.0, 0.5, 0.0 ),
                 ],    
    edges = ('line3', [ (0,3,1), (1,4,2), (2,5,0) ], ),
    reversed = (2,1,0,4,3,5),
    drawfaces = [('tri3', [ (0,3,5),(3,1,4),(4,2,5),(3,4,5) ] )]
)

Quad4 = createElementType(
    'quad4',"A 4-node quadrilateral",
    ndim = 2,
    vertices = [ (  0.0,  0.0, 0.0 ),
                 (  1.0,  0.0, 0.0 ),
                 (  1.0,  1.0, 0.0 ),
                 (  0.0,  1.0, 0.0 ),
                 ],
    edges = ('line2', [ (0,1), (1,2), (2,3), (3,0) ], ),
    )

Quad6 = createElementType(
    'quad6',"A 6-node quadrilateral",
    ndim = 2,
    vertices = Coords.concatenate([
        Quad4.vertices,
        [ (  0.5,  0.0, 0.0 ),
          (  0.5,  1.0, 0.0 ),
          ]]),
    edges = ('line3', [ (0,4,1), (1,1,2), (2,5,3), (3,3,0) ] ),
    reversed = (3,2,1,0,5,4),
    drawedges = [ ('line2', [(1,2), (3,0)]), 
                  ('line3', [(0,4,1), (2,5,3)])
                  ],
#    drawfaces = [('tri3',[(0,4,3),(4,5,3),(4,1,5),(1,2,5)])]
    drawfaces = [('quad4',[(0,4,5,3),(2,5,4,1)], )],
    )

Quad8 = createElementType(
    'quad8',"A 8-node quadrilateral",
    ndim = 2,
    vertices = Coords.concatenate([
        Quad4.vertices,
        [ (  0.5,  0.0, 0.0 ),
          (  1.0,  0.5, 0.0 ),
          (  0.5,  1.0, 0.0 ),
          (  0.0,  0.5, 0.0 ),
          ]]),
    edges = ('line3',[ (0,4,1), (1,5,2), (2,6,3), (3,7,0), ]),
    reversed = (3,2,1,0,6,5,4,7),
#    drawfaces = [('tri3', [(0,4,7), (1,5,4), (2,6,5), (3,7,6), (4,5,6), (4,6,7) ], )],
    drawfaces = [('tri3', [(0,4,7), (1,5,4), (2,6,5), (3,7,6)]), ('quad4', [(4,5,6,7)], )],
    drawfaces2 = [('quad8', [(0,1,2,3,4,5,6,7)], )],
    )
    

Quad9 = createElementType(
    'quad9',"A 9-node quadrilateral",
    ndim = 2,
    vertices = Coords.concatenate([
        Quad8.vertices,
        [ (  0.5,  0.5, 0.0 ),
          ]]),
    edges = Quad8.edges,
    reversed = (3,2,1,0,6,5,4,7,8),
#    drawfaces = [('tri3', [(0,4,8),(4,1,8),(1,5,8),(5,2,8),(2,6,8),(6,3,8),(3,7,8),(7,0,8) ], )],
    drawfaces = [('quad4', [(0,4,8,7),(1,5,8,4),(2,6,8,5),(3,7,8,6) ], )],
    drawfaces2 = [('quad9', [(0,1,2,3,4,5,6,7,8)], )],
    )

######### 3D ###################

Tet4 = createElementType(
    'tet4',"A 4-node tetrahedron",
    ndim = 3,
    vertices = [ ( 0.0, 0.0, 0.0 ),
                 ( 1.0, 0.0, 0.0 ),
                 ( 0.0, 1.0, 0.0 ),
                 ( 0.0, 0.0, 1.0 ),
                 ],
    edges = ('line2', [ (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) ], ), 
    faces = ('tri3', [ (0,2,1), (0,1,3), (1,2,3), (2,0,3) ], ),
    reversed = (0,1,3,2),
    )


Tet10 = createElementType(
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
                 ],
    edges = ('line3', [ (0,4,1),(1,7,2),(2,5,0),(0,6,3),(1,9,3),(2,8,3) ],), 
    # BV: This needs further specification!
    faces = Tet4.faces,
    reversed = (0,1,3,2,4,6,5,9,8,7),
    )


Tet14 = createElementType(
    'tet14',"A 14-node tetrahedron",
    ndim = 3,
    vertices = Coords.concatenate([
        Tet10.vertices,
        [ ( 1./3., 1./3., 0.0 ),
          ( 0.0, 1./3., 1./3. ),
          ( 1./3., 0.0, 1./3. ),
          ( 1./3., 1./3., 1./3. ),
          ]]),
    edges = Tet10.edges, 
    # BV: This needs further specification!
    faces = Tet4.faces,
    reversed = (0,1,3,2,4,6,5,9,8,7,12,11,10,13),
    )
    
    
Tet15 = createElementType(
    'tet15',"A 15-node tetrahedron",
    ndim = 3,
    vertices = Coords.concatenate([
        Tet14.vertices,
        [ ( 0.25, 0.25, 0.25 ),
          ]]),
    edges = Tet10.edges,
    # BV: This needs further specification!
    faces = Tet4.faces,
    reversed = (0,1,3,2,4,6,5,9,8,7,12,11,10,13,14),
    )


Wedge6 = createElementType(
    'wedge6',"A 6-node wedge element",
    ndim = 3,
    vertices = Coords.concatenate([
        Tri3.vertices,
        [ ( 0.0, 0.0, 1.0 ),
          ( 1.0, 0.0, 1.0 ),
          ( 0.0, 1.0, 1.0 ),
          ]]),
    edges = ('line2', [ (0,1), (1,2), (2,0), (0,3), (1,4), (2,5), (3,4), (4,5), (5,3) ], ),
    faces = ('quad4', [ (0,2,2,1), (3,4,4,5), (0,1,4,3), (1,2,5,4), (0,3,5,2) ], ),
    reversed = (3,4,5,0,1,2),
    drawfaces = [ ('tri3', [ (0,2,1), (3,4,5)] ),
                  ('quad4', [(0,1,4,3), (1,2,5,4), (0,3,5,2) ], )]
    )


Hex8 = createElementType(
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
    edges = ('line2',[ (0,1), (1,2), (2,3), (3,0),
                       (4,5), (5,6), (6,7), (7,4),
                       (0,4), (1,5), (2,6), (3,7) ], ),
    faces = ('quad4', [ (0,4,7,3), (1,2,6,5),
                        (0,1,5,4), (3,7,6,2),
                        (0,3,2,1), (4,5,6,7) ], ),
    reversed = (4,5,6,7,0,1,2,3),
    )



Hex16 = createElementType(
    'hex16',"A 16-node hexahedron",
    ndim = 3,
    vertices = Coords.concatenate([
        Hex8.vertices,
        [(  0.5,  0.0, 0.0 ),
         (  1.0,  0.5, 0.0 ),
         (  0.5,  1.0, 0.0 ),
         (  0.0,  0.5, 0.0 ),
         (  0.5,  0.0, 1.0 ),
         (  1.0,  0.5, 1.0 ),
         (  0.5,  1.0, 1.0 ),
         (  0.0,  0.5, 1.0 ),
         ]]),
    edges = ('line3', [ (0,8,1), (1,9,2), (2,10,3),(3,11,0),
                        (4,12,5),(5,13,6),(6,14,7),(7,15,4),
                        (0,0,4),(1,1,5),(2,2,6),(3,3,7) ], ),
    faces = ('quad8', [ (0,4,7,3,0,15,7,11), (1,2,6,5,9,2,13,5),
                        (0,1,5,4,8,1,12,4), (3,7,6,2,3,14,6,10), 
                        (0,3,2,1,11,10,9,8), (4,5,6,7,12,13,14,15) ], ),
    reversed= (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11),
    drawedges = [ Hex8.edges ],
    drawfaces = [ Hex8.faces ]
    )


Hex20 = createElementType(
    'hex20',"A 20-node hexahedron",
    ndim = 3,
    vertices = Coords.concatenate([
        Hex16.vertices,
        [(  0.0,  0.0, 0.5 ),
         (  1.0,  0.0, 0.5 ),
         (  1.0,  1.0, 0.5 ),
         (  0.0,  1.0, 0.5 )
         ]]),
    edges = ('line3',[ (0,8,1), (1,9,2), (2,10,3),(3,11,0),
                       (4,12,5),(5,13,6),(6,14,7),(7,15,4),
                       (0,16,4),(1,17,5),(2,18,6),(3,19,7) ],),
    faces = ('quad8',[ (0,4,7,3,16,15,19,11), (1,2,6,5,9,18,13,17),
                       (0,1,5,4,8,17,12,16), (3,7,6,2,19,14,18,10),
                       (0,3,2,1,11,10,9,8), (4,5,6,7,12,13,14,15) ], ),
    reversed = (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,16,17,18,19),
)

Hex20.drawfaces = [ Hex20.faces.selectNodes(i) for i in Quad8.drawfaces ]
Hex20.drawfaces2 = [ Hex20.faces ]


# THIS ELEMENT USES A REGULAR NODE NUMBERING!!
# WE MIGHT SWITCH OTHER ELEMENTS TO THIS REGULAR SCHEME TOO
# AND ADD THE RENUMBERING TO THE FE OUTPUT MODULES
from simple import regularGrid
Hex27 = createElementType(
    'hex27',"A 27-node hexahedron",
    ndim = 3,
    vertices = regularGrid([0.,0.,0.],[1.,1.,1.],[2,2,2]).swapaxes(0,2).reshape(-1,3),
    edges = ('line3',[ (0,1,2),(6,7,8),(18,19,20),(24,25,26),
                       (0,3,6),(2,5,8),(18,21,24),(20,23,26),
                       (0,9,18),(2,11,20),(6,15,24),(8,17,26) ],),
    faces = ('quad9',[ (0,18,24,6,9,21,15,3,12),(2,8,26,20,5,17,23,11,14),
                       (0,2,20,18,1,11,19,9,10),(6,24,26,8,15,25,17,7,16),
                       (0,6,8,2,3,7,5,1,4),(18,20,26,24,19,23,25,21,22), ],),
)
Hex27.drawfaces = [ Hex27.faces.selectNodes(i) for i in Quad9.drawfaces ]

######################################################################
########## element type conversions ##################################

_conversion_doc_ = """Element type conversion

Element type conversion in pyFormex is a powerful feature to transform
Mesh type objects. While mostly used to change the element type, there
are also conversion types that refine the Mesh.

Available conversion methods are defined in an attribute `conversion`
of the input element type. This attribute should be a dictionary, where
the keys are the name of the conversion method and the values describe
what steps need be taken to achieve this conversion. The method name
should be the name of the target element, optionally followed by a suffix
to discriminate between different methods yielding the same target element type.
The suffix should always start with a '-'. The part starting at the '-' will
be stripped of to set the final target element name.

E.g., a 'line3' element type is a quadratic line element through three points.
There are two available methods to convert it to 'line2' (straight line
segments betwee two points), named named 'line2', resp. 'line2-2'.
The first will transform a 'line3' element in a single 'line2' between
the two endpoints (i.e. the chord of the quadratic element);
the second will replace each 'line3' with two straight segments: from
first to central node, and from central node to end node (i.e. the tangents).

The values in the dictionary are a list of execution steps to be performed
in the conversion. Each step is a tuple of a single character defining the
type of the step, and the data needed by this type of step. The steps are
executed one by one to go from the source element type to the target.

Currently, the following step types are defined:

==============  =============================================
   Type         Data
==============  =============================================
's' (select)    connectivity list of selected nodes
'a' (average)   list of tuples of nodes to be averaged
'v' (via)       string with name of intermediate element type
'f' (function)  a proper conversion function
'r' (random)    list of conversion method names
==============  =============================================

The operation of these methods is as follows:

:'s' (select): This is the most common conversion type. It selects a set of
  nodes of the input element, and creates one or more new elements with these
  nodes. The data field is a list of tuples defining for each created element
  which node numbers from the source element should be included. This method
  will usually decrease the plexitude of the elements.

:'a' (average): Creates new nodes the position of which is computed as
  an average of existing nodes. The data field is a list of tuples with
  the numbers of the nodes that should be averaged for each new node. The
  resulting new nodes are added in order at the end of the existing nodes.
  If this order is not the proper local node numbering, an 's' step should
  follow to put the (old and new) nodes in the proper order.
  This method will usually increase the plexitude of the elements.

:'v' (via): The conversion is made via an intermediate element type. The source
  Mesh is first converted to this intermediate type, and the result is then
  transformed to the target type.

:r' (random): Chooses a random method between a list of alternatives. The data
  field is a list of conversion method names defined for the same element
  (and thus inside the same dictionary). While this could be considered
  an amusement (e.g. used in the Carpetry example), there are serious
  application for this, e.g. when transforming a Mesh of squares or rectangles
  into a Mesh of triangles, by adding one diagonal in each element.
  Results with such a Mesh may however be different dependent on the choice
  of diagonal. The converted Mesh has directional preferences, not present
  in the original. The Quad4 to Tri3 conversion therefore has the choice to
  use either 'up' or 'down' diagonals. But a better choice is often the
  'random' method, which will put the diagonals in a random direction, thus
  reducing the effect.

"""

Line3.conversions = {
    'line2'   : [ ('s', [ (0,2) ]), ],
    'line2-2' : [ ('s', [ (0,1), (1,2) ]), ],
    }
Tri3.conversions =  {
    'tri3-4' : [ ('v', 'tri6'), ],
    'tri6'   : [ ('a', [ (0,1), (1,2), (2,0) ]), ],
    'quad4'  : [ ('v', 'tri6'), ],
    }
Tri6.conversions = {
    'tri3'   : [ ('s', [ (0,1,2) ]), ],
    'tri3-4' : [ ('s', [ (0,3,5),(3,1,4),(4,2,5),(3,4,5) ]), ],
    'quad4'  : [ ('a', [ (0,1,2), ]),
                 ('s', [ (0,3,6,5),(1,4,6,3),(2,5,6,4) ]),
                 ],
    }
Quad4.conversions = {
    'tri3'   : 'tri3-u',
    'tri3-r' : [ ('r', ['tri3-u','tri3-d']), ],
    'tri3-u' : [ ('s', [ (0,1,2), (2,3,0) ]), ],
    'tri3-d' : [ ('s', [ (0,1,3), (2,3,1) ]), ],
    'tri3-x' : [ ('a', [ (0,1,2,3) ]),
                 ('s', [ (0,1,4),(1,2,4),(2,3,4),(3,0,4) ]),
                 ],
    'quad8'  : [ ('a', [ (0,1), (1,2), (2,3), (3,0) ])],
    'quad4-4': [ ('v', 'quad9'), ],
    'quad9'  : [ ('v', 'quad8'), ],
    }
Quad6.conversions = {
    'quad8'  : [ ('a',[ (0,3), (1,2)]),
                 ('s',[(0, 1, 2, 3, 4, 7, 5, 6)])],
    'quad9'  : [ ('a',[ (0,3), (1,2), (4,5)]), ],
    }
Quad8.conversions = {
    'tri3'   : [ ('v', 'quad9'), ],
    'tri3-v' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(5,6,4),(7,4,6) ]), ],
    'tri3-h' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(4,5,7),(6,7,5) ]), ],
    'quad4'  : [ ('s', [ (0,1,2,3) ]), ],
    'quad4-4': [ ('v', 'quad9'), ],
    'quad9'  : [ ('a', [ (4,5,6,7) ]), ],
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
    'tet10'  : [ ('a', [ (0,1), (0,2), (0,3), (1,2), (2, 3), (1, 3)]), ],
    'tet14'  : [ ('v', 'tet10'), ],
    'tet15'  : [ ('v', 'tet14'), ],
    'hex8'   : [ ('v', 'tet15'), ],
    }
Tet10.conversions = {
    'tet4'   :  [ ('s', [ (0,1,2,3,) ]), ],
    'tet14'  : [ ('a', [ (0,1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3), ]), ],
    'tet15'  : [ ('v', 'tet14'), ],
    'hex8'   : [ ('v', 'tet15'), ],
    }
Tet14.conversions = {
    'tet10' : [ ('s', [ (0,1,2,3,4, 5, 6, 7, 8, 9) ]), ],
    'tet4'  : [ ('v', 'tet10'), ],
    'tet15' : [ ('a', [ (0,1, 2, 3), ]), ],
    'hex8'  : [ ('v', 'tet15'), ],
    }
Tet15.conversions = {
    'tet14' :  [ ('s', [ (0,1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13) ]), ],
    'tet10' :  [ ('v', 'tet14'), ],
    'tet4'  :  [ ('v', 'tet10'), ],
    'hex8'  :  [ ('s', [ (0,4,10, 5, 6, 12, 14, 11), (4,1,7, 10, 12, 9, 13, 14),
                       (5, 10, 7,2,11, 14, 13, 8), (6, 12, 14, 11, 3, 9, 13, 8) ]), ],
    }
Wedge6.conversions = {
    'tet4'    : 'tet4-11',
    'tet4-3'  : [ ('s', [ (0,1,2,3),(1,2,3,4),(2,3,4,5) ]), ],
    'tet4-11' : [ ('a', [ (0,1, 4, 3), (1,2, 5, 4), (0, 2, 5, 3)]),
                  ('s', [   (0, 1, 2, 6), (0, 6, 2, 8), (1, 2, 6, 7), (2, 6, 7, 8), (1, 7, 6, 4), (0, 6, 8, 3), (2, 8, 7, 5), (6,7, 8, 3), (3,7, 8, 5), (3,4, 7, 5), (3,6, 7, 4)  ]), ],
    }
Hex8.conversions = {
    'wedge6' : [ ('s', [ (0,1,2,4,5,6),(2,3,0,6,7,4) ]), ],
    'tet4'   : 'tet4-24',
    'tet4-5' : [ ('s', [ (0,1,2,5),(2,3,0,7),(5,7,6,2),(7,5,4,0),(0,5,2,7) ]), ],
    'tet4-6' : [ ('v', 'wedge6') ],
    'tet4-24': [ ('a', [(0,3,2,1),(0,1,5,4),(0,4,7,3),(1,2,6,5),(2,3,7,6),(4,5,6,7)]),
                 ('a', [(0,1,2,3,4,5,6,7)]), 
                 ('s', [(0,1,8,14),(1,2,8,14),(2,3,8,14),(3,0,8,14), 
                        (0,4,9,14),(4,5,9,14),(5,1,9,14),(1,0,9,14), 
                        (0,3,10,14),(3,7,10,14),(7,4,10,14),(4,0,10,14),
                        (1,5,11,14),(5,6,11,14),(6,2,11,14),(2,1,11,14),
                        (2,6,12,14),(6,7,12,14),(7,3,12,14),(3,2,12,14),    
                        (4,7,13,14),(7,6,13,14),(6,5,13,14),(5,4,13,14),]),], 
    'hex8-8': [ ('v', 'hex20'), ],
    'hex20' : [ ('a', [ (0,1), (1,2), (2,3), (3,0),
                        (4,5), (5,6), (6,7), (7,4),
                        (0,4), (1,5), (2,6), (3,7), ]), ],
    }
Hex16.conversions = {
    'hex20'  : [ ('a',[ (0,8), (1,9), (2,10), (3,11) ]),
                 ('s',[(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19)])],
    }
Hex20.conversions = {
    'hex8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
    'hex8-8': [ ('v', 'hex27'), ],
    'hex27' : [ ('a', [ (0,1,2,3),(0,1,5,4),(0,3,7,4),(1,2,6,5),(2,6,7,3),(4,5,6,7), ]), 
                ('a', [ (0,1,2,3,4,5,6,7), ]),                                                                                          
                ('s', [ (0,8,1,11,20,9,3,10,2,16,21,17,22,26,23,19,24,18,4,12,5,15,25,13,7,14,6), ]), 
                ],
    'tet4'  : [ ('v', 'hex8'), ],
    }
Hex27.conversions = {
    'hex8-8': [ ('s', [ (0, 1, 4, 3, 9, 10,13,12), 
                        (1, 2, 5, 4, 10,11,14,13), 
                        (3, 4, 7, 6, 12,13,16,15), 
                        (4, 5, 8, 7, 13,14,17,16), 
                        (9, 10,13,12,18,19,22,21), 
                        (10,11,14,13,19,20,23,22), 
                        (12,13,16,15,21,22,25,24), 
                        (13,14,17,16,22,23,26,25), 
                      ]), ],
    }

##########################################################
############ Extrusions ##################################
#
# Extrusion database
#
# For each element, extruded is a dictionary with
#
#    key = degree of extrusion (1 or 2)
#  value = tuple (Target element type, Node reordering)
# If no Node reordering is specified, the nodes of the translated entity
# are jus append to those of the original entity.
#
# NEED TO CHECK THIS !!!!:
# For degree 2, the default order is:
#   first plane, intermediate plane, last plane.

Point.extruded = { 1: (Line2, []),
                   2: (Line3, [0,2,1]) }
Line2.extruded = { 1: (Quad4, [0,1,3,2] ) }
Line3.extruded = { 1: (Quad6, [0,2,5,3,1,4]),
                   2: (Quad9, [0,1,7,6,2,4,8,3,5]), }
Tri3.extruded = { 1: (Wedge6, [] ) }
Quad4.extruded = { 1: (Hex8, [] ) }
Quad8.extruded = { 1: (Hex16, [0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15] ),
                   2: (Hex20, [0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23,8,9,10,11] ) }
# BV: If Quad9 would be numbered consecutively, extrusion would be as easy as
#Quad9.extruded = { 2: (Hex27, [] }
Quad9.extruded = { 2: (Hex27, [ 0, 4, 1, 7, 8, 5, 3, 6, 2,
                                9,13,10,16,17,14,12,15,11,
                               18,22,19,25,26,23,21,24,20,
                                ]) }

############################################################
############ Reduction of degenerate elements ##############

Line3.degenerate = {
    'line2' : [ ([[0,1]], [0,2]),
                ([[1,2]], [0,2]),
                ([[0,2]], [0,1]),
                ],
    }
#
#  TODO: Are these still correct after change of wedge6?
#

Hex8.degenerate = {
    'wedge6' : [ ([[0,1],[4,5]], [0,2,3,4,6,7]),
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
                 ],
    }


##########################################################  
# This element added just for fun, no practical importance

from arraytools import golden_ratio as phi

Icosa = createElementType(
    'icosa',
    """An icosahedron: a regular polyhedron with 20 triangular surfaces.,

    nfaces = 20, nedges = 30, nvertices = 12

    All points of the icosahedron lie on a sphere with unit radius.  
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
    edges = ('line2', [ (0,1),  (0,8), (1,8), (0,10),(1,10),
                        (2,3),  (2,9), (3,9), (2,11),(3,11),
                        (4,5),  (4,0), (5,0), (4,2), (5,2),
                        (6,7),  (6,1), (7,1), (6,3), (7,3),
                        (8,9),  (8,4), (9,4), (8,6), (9,6),
                        (10,11),(10,5),(11,5),(10,7),(11,7),
                        ], ),
    faces = ('tri3', [ (0,1,8),  (1,0,10),
                       (2,3,11), (3,2,9),
                       (4,5,0),  (5,4,2),
                       (6,7,3),  (7,6,1),
                       (8,9,4),  (9,8,6),
                       (10,11,7),(11,10,5),
                       (0,8,4),  (1,6,8),
                       (0,5,10), (1,10,7),
                       (2,11,5), (3,7,11),
                       (2,4,9),  (3,9,6),
                       ], ),
    reversed = (2,3,0,1,4,5,6,7,9,8,11,10),
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

    Returns: a subclass of :class:`ElementType`

    Errors: if neither `name` nor `nplex` can resolve into an element type,
      an error is raised.

    Example:
    
    >>> elementType('tri3').name()
    'tri3'
    >>> elementType(nplex=2).name()
    'line2'
    """
    eltype = name

    try:
        if issubclass(eltype,ElementType):
            return eltype
    except:
        pass
    
    if eltype is None:
        try:
            return _default_eltype[nplex]
        except:
            pass

    try:
        eltype = globals()[name.capitalize()]
        if issubclass(eltype,ElementType):
            return eltype
    except:
        pass

    #raise ValueError,"No such element type: %s" % name
    return None


def elementTypes(ndim=None):
    """Return the names of available elements.

    If a value is specified for ndim, only the elements with the matching
    dimensionality are returned.
    """
    if ndim is None:
        return _registered_element_types.keys()
    else:
        return [ k for k,v in _registered_element_types.iteritems() if v.ndim==ndim] 


def printElementTypes():
    """Print all available element types.

    Prints a list of the names of all available element types,
    grouped by their dimensionality.
    """
    print("Available Element Types:")        
    for ndim in range(4):
        print("  %s-dimensional elements: %s" % (ndim,elementTypes(ndim))        )

if __name__ == "__main__":
    printElementTypes()

# End
