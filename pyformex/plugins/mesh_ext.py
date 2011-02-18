# $Id$

"""Extended functionality of the Mesh class.

This module defines extended Mesh functionality which is considered to be
experimental, maybe incomplete or even buggy.

The functions in this module can be called as functions operating on a
Mesh object, but can also be installed as Mesh methods by calling the
initialize() function once.
"""

from mesh import Mesh
from elements import elementType
from trisurface import areaNormals
from formex import *
from connectivity import Connectivity

##############################################################################
#
# These functions replace the standard Mesh.report method.
# They are here for demonstration purposes.
#
def report(self):
    """Create a report on the Mesh shape and size.

    The report contains the number of nodes, number of elements,
    plexitude, element type, bbox and size.
    """
    bb = self.bbox()
    return """
    
Shape: %s nodes, %s elems, plexitude %s
Eltype: %s (ndim=%s),
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),self.eltype,self.eltype.ndim,bb[0],bb[1],bb[1]-bb[0])


def alt_report(self):
    """Create a report on the Mesh shape and size.

    The report contains the number of nodes, number of elements,
    plexitude, element type, bbox and size.
    """
    bb = self.bbox()
    return """
    
Number of nodes: %s
Number of elements: %s
Plexitude: %s
Eltype: %s
Dimensionality: %s
Min Coords: %s
Max Coords: %s
Size: %s
Area: %s
Volume: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),self.eltype,self.eltype.ndim,bb[1],bb[0],bb[1]-bb[0],self.area(),self.volume())

##############################################################################
#
# Define other functions here
#


##GDS: needed for function ring.
##remove if it is already in pyFormex
def removeAllDoubles(hi):
    """
    It returns only values that appear only once.
    
    hi is an array of integers.
    """
    hiinv = Connectivity(hi).inverse()
    ncon = (hiinv>=0).sum(axis=1)
    return arange(len(ncon))[ncon==1].reshape(-1) 


##GDS: needed for function ring.
##remove if it is already in pyFormex
def removeInteg(a, b):
    """
    Remove the values of b that are found in a.
    
    a and b are two arrays of integers.   
    a=array([1, 2, 4, 5, 3, 6])
    b=array([1, 4, 3, 7])
    removeInt(a, b)
    array([2,5,6])
    """
    a, b=array(a), array(b)
    ib= matchIndex(a, b)#in case b has values that are not in a
    c=b[ib!=-1]
    hi=append(a, c)
    return removeAllDoubles(hi)


def rings(adj, sources, step=1):
    """
    It finds the rings of elems connected to sources by node.
    
    Sources is a list of elem indices.
    adj is the adjacency table and should be calulated before as
    adj=mesh.elems.adjacency(kind='e')
    A list of rings is returned, from zero (equal to sources) to step.
    If step is None, all rings are returned.
    """

    R=[sources]
    if step is None:
        step=len(adj)
    for i in range(step):
        newring=unique(adj[ R[-1] ])[1:]
        R.append(removeInteg(newring,  concatenate(R) ))
        if len(R[-1])==0:
            return R[:-1]
    return R

## THIS NEEDS WORK ###
## surfacetype is also eltype ??

def areas(self):
    """area of elements

    For surface element the faces' area is returned.
    For volume elements the sum of the faces'areas is returned.

    """

    #In case of quadratic faces, the face's area should be 
    #the area inside the polygon of face vertices or 
    #the area of the equivalent linear face?

    ##this function would require some changes (here proposed inside the function as starting):
    ##create a _default_surfacetype to create quad8 instead of hex8 ?maybe also a _default_volumetype to create tet4 instead of quad4 ?

    _default_surfacetype = {
    3 : 'tri3',
    4 : 'quad4',
    6 : 'tri6',
    8 : 'quad8',
    9 : 'quad9',
    }
    def defaultSurfacetype(nplex):
        """Default face type for a surface mesh with given plexitude.

        For the most common cases of plexitudes, we define a default face
        type. The full list of default types can be found in
        mesh._default_facetype.
        """
        return _default_surfacetype.get(nplex,None)

    nfacperel= len(self.eltype.faces)#nfaces per elem
    mf=Mesh(self.coords, self.getFaces())#mesh of all faces
    mf.eltype = elementType(defaultSurfacetype(mf.nplex()))
    ntriperfac= mf.select([0]).convert('tri3').nelems()#how many tri per face
    elfacarea= areaNormals( mf.convert('tri3').toFormex()[:])[0].reshape(self.nelems(), nfacperel*ntriperfac)#elems'faces'areas
    return elfacarea.sum(axis=1)#elems'areas


def area(self):
    """Return the total area of the Mesh.

    For a Mesh with dimensionality 2, the total area of the Mesh is returned.
    For a Mesh with dimensionality 3, the total area of all the element faces
    is returned. Use Mesh.getBorderMesh().area() if you only want the total
    area of the border faces.
    For a Mesh with dimensionality < 2, 0 is returned.
    """
    try:
        return self.areas().sum()
    except:
        return 0.0


##############################################################################
#
# Initialize
#

def initialize():
    """Initialize the Mesh extensions.

    Calling this function will install some of the mesh functions
    defined in this modules as Mesh methods.
    """
    Mesh.report = alt_report


def _auto_initialize():
    """Auto-initialize Mesh extensions.

    Calling this function will install some of the mesh functions
    defined in this modules as Mesh methods.
    This function is called when the module is loaded, so the functions
    installed here will always be available as Mesh methods just by
    importing the mesh_ext module.
    """
    Mesh.report = report
    Mesh.areas = areas
    Mesh.area = area
    

_auto_initialize()


# End
