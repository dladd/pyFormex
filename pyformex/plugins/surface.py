# $Id$
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
"""Operations on triangulated surfaces.

A triangulated surface is a surface consisting solely of triangles.
Any surface in space, no matter how complex, can be approximated with
a triangulated surface.
"""

import os
import pyformex as GD
from plugins import tetgen
from plugins.mesh import Mesh
from connectivity import *
from utils import runCommand, changeExt,countLines,mtime,hasExternal
from formex import *
import tempfile
from numpy import *
from gui.drawable import interpolateNormals
from plugins.geomtools import projectionVOP,rotationAngle
from plugins import inertia

hasExternal('admesh')
hasExternal('tetgen')
hasExternal('gts')



def areaNormals(x):
    """Compute the area and normal vectors of a collection of triangles.

    x is an (ntri,3,3) array of coordinates.

    Returns a tuple of areas,normals.
    The normal vectors are normalized.
    The area is always positive.
    """
    area,normals = vectorPairAreaNormals(x[:,1]-x[:,0],x[:,2]-x[:,1])
    area *= 0.5
    return area,normals


# Conversion of surface file formats

def stlConvert(stlname,outname=None,options='-d'):
    """Transform an .stl file to .off or .gts format.

    If outname is given, it is either '.off' or '.gts' or a filename ending
    on one of these extensions. If it is only an extension, the stlname will
    be used with extension changed.

    If the outname file exists and its mtime is more recent than the stlname,
    the outname file is considered uptodate and the conversion programwill
    not be run.
    
    The conversion program will be choosen depending on the extension.
    This uses the external commands 'admesh' or 'stl2gts'.

    The return value is a tuple of the output file name, the conversion
    program exit code (0 if succesful) and the stdout of the conversion
    program (or a 'file is already uptodate' message).
    """
    if not outname:
        outname = GD.cfg.get('surface/stlread','.off')
    if outname.startswith('.'):
        outname = changeExt(stlname,outname)
    if os.path.exists(outname) and mtime(stlname) < mtime(outname):
        return outname,0,"File '%s' seems to be up to date" % outname
    
    if outname.endswith('.off'):
        cmd = "admesh %s --write-off '%s' '%s'" % (options,outname,stlname)
    elif outname.endswith('.gts'):
        cmd = "stl2gts < '%s' > '%s'" % (stlname,outname)
    else:
        return outname,1,"Can not convert file '%s' to '%s'" % (stlname,outname)
       
    sta,out = runCommand(cmd)
    return outname,sta,out


# Input of surface file formats

def read_gts(fn):
    """Read a GTS surface mesh.

    Return a coords,edges,faces tuple.
    """
    GD.message("Reading GTS file %s" % fn)
    fil = file(fn,'r')
    header = fil.readline().split()
    ncoords,nedges,nfaces = map(int,header[:3])
    if len(header) >= 7 and header[6].endswith('Binary'):
        sep=''
    else:
        sep=' '
    coords = fromfile(fil,dtype=Float,count=3*ncoords,sep=' ').reshape(-1,3)
    edges = fromfile(fil,dtype=int32,count=2*nedges,sep=' ').reshape(-1,2) - 1
    faces = fromfile(fil,dtype=int32,count=3*nfaces,sep=' ').reshape(-1,3) - 1
    GD.message("Read %d coords, %d edges, %d faces" % (ncoords,nedges,nfaces))
    if coords.shape[0] != ncoords or \
       edges.shape[0] != nedges or \
       faces.shape[0] != nfaces:
        GD.message("Error while reading GTS file: the file is probably incorrect!")
    return coords,edges,faces


def read_off(fn):
    """Read an OFF surface mesh.

    The mesh should consist of only triangles!
    Returns a nodes,elems tuple.
    """
    GD.message("Reading .OFF %s" % fn)
    fil = file(fn,'r')
    head = fil.readline().strip()
    if head != "OFF":
        print("%s is not an OFF file!" % fn)
        return None,None
    nnodes,nelems,nedges = map(int,fil.readline().split())
    nodes = fromfile(file=fil, dtype=Float, count=3*nnodes, sep=' ')
    # elems have number of vertices + 3 vertex numbers
    elems = fromfile(file=fil, dtype=int32, count=4*nelems, sep=' ')
    GD.message("Read %d nodes and %d elems" % (nnodes,nelems))
    return nodes.reshape((-1,3)),elems.reshape((-1,4))[:,1:]


def read_stl(fn,intermediate=None):
    """Read a surface from .stl file.

    This is done by first coverting the .stl to .gts or .off format.
    The name of the intermediate file may be specified. If not, it will be
    generated by changing the extension of fn to '.gts' or '.off' depending
    on the setting of the 'surface/stlread' config setting.
    
    Return a coords,edges,faces or a coords,elems tuple, depending on the
    intermediate format.
    """
    ofn,sta,out = stlConvert(fn,intermediate)
    if sta:
        GD.debug("Error during conversion of file '%s' to '%s'" % (fn,ofn))
        GD.debug(out)
        return ()

    if ofn.endswith('.gts'):
        return read_gts(ofn)
    elif ofn.endswith('.off'):
        return read_off(ofn)


def read_gambit_neutral(fn):
    """Read a triangular surface mesh in Gambit neutral format.

    The .neu file nodes are numbered from 1!
    Returns a nodes,elems tuple.
    """
    runCommand("/home/pmortier/pyformex/external/gambit-neu '%s'" % fn)
    nodesf = changeExt(fn,'.nodes')
    elemsf = changeExt(fn,'.elems')
    nodes = fromfile(nodesf,sep=' ',dtype=Float).reshape((-1,3))
    elems = fromfile(elemsf,sep=' ',dtype=int32).reshape((-1,3))
    return nodes, elems-1


# Output of surface file formats

def write_gts(fn,nodes,edges,faces):
    if nodes.shape[1] != 3 or edges.shape[1] != 2 or faces.shape[1] != 3:
        raise runtimeError, "Invalid arguments or shape"
    fil = file(fn,'w')
    fil.write("%d %d %d\n" % (nodes.shape[0],edges.shape[0],faces.shape[0]))
    for nod in nodes:
        fil.write("%s %s %s\n" % tuple(nod))
    for edg in edges+1:
        fil.write("%d %d\n" % tuple(edg))
    for fac in faces+1:
        fil.write("%d %d %d\n" % tuple(fac))
    fil.write("#GTS file written by %s\n" % GD.Version)
    fil.close()


def write_stla(f,x):
    """Export an x[n,3,3] float array as an ascii .stl file."""

    own = type(f) == str
    if own:
        f = file(f,'w')
    f.write("solid  Created by %s\n" % GD.Version)
    area,norm = areaNormals(x)
    degen = degenerate(area,norm)
    print("The model contains %d degenerate triangles" % degen.shape[0])
    for e,n in zip(x,norm):
        f.write("  facet normal %s %s %s\n" % tuple(n))
        f.write("    outer loop\n")
        for p in e:
            f.write("      vertex %s %s %s\n" % tuple(p))
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write("endsolid\n")
    if own:
        f.close()


def write_stlb(f,x):
    """Export an x[n,3,3] float array as an binary .stl file."""
    print("Cannot write binary STL files yet!" % fn)
    pass


def write_gambit_neutral(fn,nodes,elems):
    print("Cannot write Gambit neutral files yet!" % fn)
    pass


def write_off(fn,nodes,elems):
    if nodes.shape[1] != 3 or elems.shape[1] < 3:
        raise runtimeError, "Invalid arguments or shape"
    fil = file(fn,'w')
    fil.write("OFF\n")
    fil.write("%d %d 0\n" % (nodes.shape[0],elems.shape[0]))
    for nod in nodes:
        fil.write("%s %s %s\n" % tuple(nod))
    format = "%d %%d %%d %%d\n" % elems.shape[1]
    for el in elems:
        fil.write(format % tuple(el))
    fil.close()


def write_smesh(fn,nodes,elems):
    tetgen.writeSurface(fn,nodes,elems)


def surface_volume(x,pt=None):
    """Return the volume inside a 3-plex Formex.

    - `x`: an (ntri,3,3) shaped float array, representing ntri triangles.
    - `pt`: a point in space. If unspecified, it is taken equal to the
      center() of the coordinates `x`.

    Returns an (ntri) shaped array with the volume of the tetraeders formed
    by the triangles and the point `pt`. If `x` represents a closed surface,
    the sum of this array will represent the volume inside the surface.
    """
    if pt is None:
        pt = x.center()
    x = x - pt
    a,b,c = [ x[:,i,:] for i in range(3) ]
    d = cross(b,c)
    e = (a*d).sum(axis=-1)
    v = sign(e) * abs(e)/6.
    return v


def curvature(coords,elems,edges,neighbours=1):
    """Calculate curvature parameters
    
    (according to Dong and Wang 2005;
    Koenderink and Van Doorn 1992).
    The n-ring neighbourhood of the nodes is used (n=neighbours).
    Eight values are returned: the Gaussian and mean curvature, the
    shape index, the curvedness, the principal curvatures and the
    principal directions.
    """
    # calculate n-ring neighbourhood of the nodes (n=neighbours)
    adj = adjacencyArray(edges,neighbours=neighbours)
    adjNotOk = adj<0
    # calculate unit length average normals at the nodes p
    # a weight 1/|gi-p| could be used (gi=center of the face fi)
    p = coords
    n = interpolateNormals(coords,elems,atNodes=True)
    # double-precision: this will allow us to check the sign of the angles    
    p = p.astype(float64)
    n = n.astype(float64)
    vp = p[adj] - p[:,newaxis,:]
    vn = n[adj] - n[:,newaxis,:]
    # where adjNotOk, set vectors = [0.,0.,0.]
    vp[adjNotOk] = 0.
    vn[adjNotOk] = 0.
    # calculate unit length projection of vp onto the tangent plane
    t = projectionVOP(vp,n[:,newaxis])
    t = normalize(t)
    # calculate normal curvature
    k = dotpr(vp,vn)/dotpr(vp,vp)
    # calculate maximum normal curvature and corresponding coordinate system
    imax = nanargmax(k,-1)
    kmax =  k[range(len(k)),imax]
    tmax = t[range(len(k)),imax]
    e1 = tmax
    e2 = cross(e1,n)
    e2 = normalize(e2)
    # calculate angles (e1,t), where adj = -1, set angle = 0
    theta,rot = rotationAngle(repeat(e1[:,newaxis],t.shape[1],1),t,angle_spec=Rad)
    # check the sign of the angles
    d =  dotpr(rot,n[:,newaxis])/(length(rot)*length(n)[:,newaxis]) # divide by length for round-off errors
    cw = isClose(d,[-1.])
    theta[cw] = -theta[cw]
    # calculate coefficients
    a = kmax
    a11 = nansum(cos(theta)**2*sin(theta)**2,-1)
    a12 = nansum(cos(theta)*sin(theta)**3,-1)
    a21 = a12
    a22 = nansum(sin(theta)**4,-1)
    a13 = nansum((k-a[:,newaxis]*cos(theta)**2)*cos(theta)*sin(theta),-1)
    a23 = nansum((k-a[:,newaxis]*cos(theta)**2)*sin(theta)**2,-1)
    denom = (a11*a22-a12**2)
    b = (a13*a22-a23*a12)/denom
    c = (a11*a23-a12*a13)/denom
    # for nodes that have only two adjacent nodes, (a11*a22-a12**2) = 0, b=c=inf
    # the curvature of these nodes should be zero
    zeroDenom = isClose(denom,[0.],atol=1.e-5)
    a[zeroDenom] = 0.
    b[zeroDenom] = 0.
    c[zeroDenom] = 0.
    a = nan_to_num(a)
    b = nan_to_num(b)
    c = nan_to_num(c)
    # calculate the Gaussian and mean curvature
    K = a*c-b**2/4
    H = (a+c)/2
    # calculate the principal curvatures and principal directions
    k1 = H+sqrt(H**2-K)
    k2 = H-sqrt(H**2-K)
    theta0 = 0.5*arcsin(b/(k2-k1))
    w = apply_along_axis(isClose,0,-b,2*(k2-k1)*cos(theta0)*sin(theta0))
    theta0[w] = pi-theta0[w]
    e1 = cos(theta0)[:,newaxis]*e1+sin(theta0)[:,newaxis]*e2
    e2 = cos(theta0)[:,newaxis]*e2-sin(theta0)[:,newaxis]*e1
    e1 = nan_to_num(e1)
    e2 = nan_to_num(e2)
    # calculate the shape index and curvedness
    S = 2./pi*arctan((k1+k2)/(k1-k2))
    S = nan_to_num(S)
    C = square((k1**2+k2**2)/2)
    return K,H,S,C,k1,k2,e1,e2


############################################################################


def surfaceInsideLoop(coords,elems):
    """Create a surface inside a closed curve defined by coords and elems.

    coords is a set of coordinates.
    elems is an (nsegments,2) shaped connectivity array defining a set of line
    segments forming a closed loop.

    The return value is coords,elems tuple where
    coords has one more point: the center of th original coords
    elems is (nsegment,3) and defines triangles describing a surface inside
    the original curve.
    """
    coords = Coords(coords)
    if coords.ndim != 2 or elems.ndim != 2 or elems.shape[1] != 2 or elems.max() >= coords.shape[0]:
        raise ValueError,"Invalid argument shape/value"
    x = coords[unique1d(elems)].center().reshape(-1,3)
    n = zeros_like(elems[:,:1]) + coords.shape[0]
    elems = concatenate([elems,n],axis=1)
    coords = Coords.concatenate([coords,x])
    return coords,elems


def fillHole(coords,elems):
    """Fill a hole surrounded by the border defined by coords and elems.
    
    Coords is a (npoints,3) shaped array of floats.
    Elems is a (nelems,2) shaped array of integers representing the border
    element numbers and must be ordered.
    """
    triangles = empty((0,3,),dtype=int)
    while shape(elems)[0] != 3:
        elems,triangle = create_border_triangle(coords,elems)
        triangles = row_stack([triangles,triangle])
    # remaining border
    triangles = row_stack([triangles,elems[:,0]])
    return triangles


def create_border_triangle(coords,elems):
    """Create a triangle within a border.
    
    The triangle is created from the two border elements with
    the sharpest angle.
    Coords is a (npoints,3) shaped array of floats.
    Elems is a (nelems,2) shaped array of integers representing
    the border element numbers and must be ordered.
    A list of two objects is returned: the new border elements and the triangle.
    """
    border = coords[elems]
    # calculate angles between edges of border
    edges1 = border[:,1]-border[:,0]
    edges2 = border[:,0]-border[:,1]
    # roll axes so that edge i of edges1 and edges2 are neighbours
    edges2 = roll(edges2,-1,0)
    len1 = length(edges1)
    len2 = length(edges2)
    inpr = diagonal(inner(edges1,edges2))
    cos = inpr/(len1*len2)
    # angle between 0 and 180 degrees
    angle = arccos(cos)/(pi/180.)
    # determine sharpest angle
    i = where(angle == angle.min())[0][0]
    # create triangle and new border elements
    j = i + 1
    n = shape(elems)[0]
    if j == n:
        j -= n
    old_edges = take(elems,[i,j],0)
    elems = delete(elems,[i,j],0)
    new_edge = asarray([old_edges[0,0],old_edges[-1,1]])
    if j == 0:
        elems = insert(elems,0,new_edge,0)
    else:
        elems = insert(elems,i,new_edge,0)
    triangle = append(old_edges[:,0],old_edges[-1,1].reshape(1),0)
    return elems,triangle



############################################################################
# The TriSurface class

class TriSurface(Mesh):
    """A class representing a triangulated 3D surface.

    The surface contains `ntri` triangles, each having 3 vertices with
    3 coordinates. The surface can be initialized from one of the following:
        
    - a (ntri,3,3) shaped array of floats
    - a Formex with plexitude 3
    - a Mesh with plexitude 3
    - an (ncoords,3) float array of vertex coordinates and
      an (ntri,3) integer array of vertex numbers
    - an (ncoords,3) float array of vertex coordinates,
      an (nedges,2) integer array of vertex numbers,
      an (ntri,3) integer array of edges numbers.

    Additionally, a keyword argument prop= may be specified to
    set property values.
    """

    def __init__(self,*args,**kargs):
        """Create a new surface."""
        Mesh.__init__(self)
        self.edges = self.faces = None
        self.areas = self.normals = None
        self.econn = self.conn = self.eadj = self.adj = None
        if hasattr(self,'edglen'):
            del self.edglen
            
        if len(args) == 0:
            return  # an empty surface
        
        if len(args) == 1:
            # argument should be a suitably structured geometry object
            # TriSurface, Mesh, Formex, Coords, ndarray, ...
            a = args[0]

            if isinstance(a,Mesh):
                if a.nplex() != 3 or a.eltype != 'tri3':
                    raise ValueError,"Only meshes with plexitude 3 and eltype 'tri3' can be converted to TriSurface!"
                Mesh.__init__(self,a.coords,a.elems,a.prop,'tri3')

            else:
                if not isinstance(a,Formex):
                    # something that can be converted to a Formex
                    try:
                        a = Formex(a)
                    except:
                        raise ValueError,"Can not convert objects of type %s to TriSurface!" % type(a)
                
                if a.nplex() != 3:
                    raise ValueError,"Expected an object with plexitude 3!"

                coords,elems = a.fuse()
                Mesh.__init__(self,coords,elems,a.prop,'tri3')

        else:
            # arguments are (coords,elems) or (coords,edges,faces)
            coords = Coords(args[0])
            if len(coords.shape) != 2:
                raise ValueError,"Expected a 2-dim coordinates array"
            
            if len(args) == 2:
                # arguments are (coords,elems)
                elems = Connectivity(args[1],nplex=3)
                Mesh.__init__(self,coords,elems,None,'tri3')
                
               
            elif len(args) == 3:
                # arguments are (coords,edges,faces)
                edges = Connectivity(args[1],nplex=2)
                
                if edges.size > 0 and edges.max() >= coords.shape[0]:
                    raise ValueError,"Some vertex number is too high"
            
                faces = Connectivity(args[2],nplex=3)
                
                if faces.max() >= edges.shape[0]:
                    raise ValueError,"Some edge number is too high"

                elems = Connectivity(compactElems(edges,faces))
                Mesh.__init__(self,coords,elems,None,'tri3')
                
                # since we have the extra data available, keep them
                self.edges = edges
                self.faces = faces

            else:
                raise RuntimeError,"Too many positional arguments"

            if 'prop' in kargs:
                self.setProp(kargs['prop'])


###########################################################################
    #
    #   Return information about a TriSurface
    #

    def nedges(self):
        return self.getEdges().shape[0]

    def nfaces(self):
        return self.getFaces().shape[0]

    def vertices(self):
        return self.coords
    
    def shape(self):
        """Return the number of points, edges, faces of the TriSurface."""
        return self.ncoords(),self.nedges(),self.nfaces()
 
    #
    # In the new implementation, TriSurface is derived from a Mesh,
    # thus the base information is (coords,elems).
    # Edges and Faces should always be retrieved with getEdges() and
    # getFaces(). coords and elems can directly be used as attributes
    # and should always be kept in a consistent state.
    #
    
    def getEdges(self):
        """Get the edges data."""
        if self.edges is None:
            self.edges,self.faces = self.elems.expand()
        return self.edges
    
    def getFaces(self):
        """Get the faces data."""
        if self.faces is None:
            self.edges,self.faces = self.elems.expand()
        return self.faces

    #
    # Changes to the geometry should by preference be done through the
    # __init__ function, to ensure consistency of the data.
    # Convenience functions are defined to change some of the data.
    #

    def setCoords(self,coords):
        """Change the coords."""
        self.__init__(coords,self.elems,prop=self.prop)

    def setElems(self,elems):
        """Change the elems."""
        self.__init__(self.coords,elems,prop=self.prop)

    def setEdgesAndFaces(self,edges,faces):
        """Change the edges and faces."""
        self.__init__(self.coords,edges,faces,prop=self.prop)


    def refresh(self):
        # The object should now always be consistent
        raise RuntimeError,"The implementation of TriSurface has changed!\n You should adopt your code to the new implementation, and no longer use 'refresh'"


    def append(self,S):
        """Merge another surface with self.

        This just merges the data sets, and does not check
        whether the surfaces intersect or are connected!
        This is intended mostly for use inside higher level functions.
        """
        coords = concatenate([self.coords,S.coords])
        elems = concatenate([self.elems,S.elems+self.ncoords()])
        ## What to do if one of the surfaces has properties, the other one not?
        ## The current policy is to use zero property values for the Surface
        ## without props
        prop = None
        if self.prop is not None or S.prop is not None:
            if self.prop is None:
                self.prop = zeros(shape=self.nelems(),dtype=Int)
            if S.prop is None:
                p = zeros(shape=S.nelems(),dtype=Int)
            else:
                p = S.prop
            prop = concatenate((self.prop,p))
        self.__init__(coords,elems,prop=prop)

       
    def copy(self):
        """Return a (deep) copy of the surface."""
        S = TriSurface(self.coords.copy(),self.elems.copy())
        if self.prop is not None:
            S.setProp(self.prop.copy())
        return S

    
    def select(self,idx,compact=True):
        """Return a TriSurface which holds only elements with numbers in ids.

        idx can be a single element number or a list of numbers or
        any other index mechanism accepted by numpy's ndarray
        By default, the vertex list will be compressed to hold only those
        used in the selected elements.
        Setting compress==False will keep all original nodes in the surface.
        """
        S = TriSurface(self.coords, self.elems[idx])
        if self.prop is not None:
            S.setProp(self.prop[idx])
        if compact:
            S.compact()
        return S

    # Some functions for offsetting a surface
    
    def pointNormals(self):
        """Compute the normal vectors in each point of a collection of triangles.
        
        The normal vector in a point is the average of the normal vectors of
        the neighbouring triangles.
        The normal vectors are normalized.
        """
        con = reverseIndex(self.getElems())
        NP = self.areaNormals()[1][con] #self.normal doesn't work here???
        w = where(con == -1)
        NP[w] = 0.
        NPA = NP.sum(axis=1)
        NPA /= sqrt((NPA*NPA).sum(axis=-1)).reshape(-1,1)
        return NPA

    def offset(self,distance=1.):
        """Offset a surface with a certain distance.
        
        All the nodes of the surface are translated over a specified distance
        along their normal vector.
        """
        NPA = self.pointNormals()
        coordsNew = self.coords + NPA*distance
        return TriSurface(coordsNew,self.getElems())
   

    @classmethod
    def read(clas,fn,ftype=None):
        """Read a surface from file.

        If no file type is specified, it is derived from the filename
        extension.
        Currently supported file types:

        - .stl (ASCII or BINARY)
        - .gts
        - .off
        - .neu (Gambit Neutral)
        - .smesh (Tetgen)
        """
        if ftype is None:
            ftype = os.path.splitext(fn)[1]  # deduce from extension
        ftype = ftype.strip('.').lower()
        if ftype == 'off':
            return TriSurface(*read_off(fn))
        elif ftype == 'gts':
            return TriSurface(*read_gts(fn))
        elif ftype == 'stl':
            return TriSurface(*read_stl(fn))
        elif ftype == 'neu':
            return TriSurface(*read_gambit_neutral(fn))
        elif ftype == 'smesh':
            return TriSurface(*tetgen.readSurface(fn))
        else:
            raise "Unknown TriSurface type, cannot read file %s" % fn


    def write(self,fname,ftype=None):
        """Write the surface to file.

        If no filetype is given, it is deduced from the filename extension.
        If the filename has no extension, the 'gts' file type is used.
        """
        if ftype is None:
            ftype = os.path.splitext(fname)[1]
        if ftype == '':
            ftype = 'gts'
        else:
            ftype = ftype.strip('.').lower()

        GD.message("Writing surface to file %s" % fname)
        if ftype == 'gts':
            write_gts(fname,self.coords,self.getEdges(),self.getFaces())
            GD.message("Wrote %s vertices, %s edges, %s faces" % self.shape())
        elif ftype in ['stl','off','neu','smesh']:
            if ftype == 'stl':
                write_stla(fname,self.coords[self.elems])
            elif ftype == 'off':
                write_off(fname,self.coords,self.elems)
            elif ftype == 'neu':
                write_gambit_neutral(fname,self.coords,self.elems)
            elif ftype == 'smesh':
                write_smesh(fname,self.coords,self.elems)
            GD.message("Wrote %s vertices, %s elems" % (self.ncoords(),self.nelems()))
        else:
            print("Cannot save TriSurface as file %s" % fname)


    @coordsmethod
    def reflect(self,*args,**kargs):
        if kargs.get('invert_normals',True) == True:
            elems = self.getElems()
            self.setElems(column_stack([elems[:,0],elems[:,2],elems[:,1]]))


####################### TriSurface Data ######################


    def avgVertexNormals(self):
        """Compute the average normals at the vertices."""
        return interpolateNormals(self.coords,self.elems,atNodes=True)


    def areaNormals(self):
        """Compute the area and normal vectors of the surface triangles.

        The normal vectors are normalized.
        The area is always positive.

        The values are returned and saved in the object.
        """
        if self.areas is None or self.normals is None:
            self.areas,self.normals = areaNormals(self.coords[self.elems])
        return self.areas,self.normals


    def facetArea(self):
        return self.areaNormals()[0]
        

    def area(self):
        """Return the area of the surface"""
        area = self.areaNormals()[0]
        return area.sum()


    def volume(self):
        """Return the enclosed volume of the surface.

        This will only be correct if the surface is a closed manifold.
        """
        x = self.coords[self.elems]
        return surface_volume(x).sum()


    def curvature(self,neighbours=1):
        """Return the curvature parameters at the nodes.
        
        The n-ring neighbourhood of the nodes is used (n=neighbours).
        Eight values are returned: the Gaussian and mean curvature, the
        shape index, the curvedness, the principal curvatures and the
        principal directions.      
        """
        curv = curvature(self.coords,self.elems,self.getEdges(),neighbours=neighbours)
        return curv
    
    
    def inertia(self):
        """Return inertia related quantities of the surface.
        
        This returns the center of gravity, the principal axes of inertia, the principal
        moments of inertia and the inertia tensor.
        """
        ctr,I = inertia.inertia(self.centroids(),mass=self.facetArea().reshape(-1,1))
        Iprin,Iaxes = inertia.principal(I,sort=True,right_handed=True)
        data = (ctr,Iaxes,Iprin,I)
        return data


    def edgeConnections(self):
        """Find the elems connected to edges."""
        if self.econn is None:
            self.econn = reverseIndex(self.getFaces())
        return self.econn
    

    def nodeConnections(self):
        """Find the elems connected to nodes."""
        if self.conn is None:
            self.conn = reverseIndex(self.elems)
        return self.conn
    

    def nEdgeConnected(self):
        """Find the number of elems connected to edges."""
        return (self.edgeConnections() >=0).sum(axis=-1)
    

    def nNodeConnected(self):
        """Find the number of elems connected to nodes."""
        return (self.nodeConnections() >=0).sum(axis=-1)


    def edgeAdjacency(self):
        """Find the elems adjacent to elems via an edge."""
        if self.eadj is None:
            nfaces = self.nfaces()
            rfaces = self.edgeConnections()
            # this gives all adjacent elements including element itself
            adj = rfaces[self.getFaces()].reshape(nfaces,-1)
            fnr = arange(nfaces).reshape(nfaces,-1)
            self.eadj = adj[adj != fnr].reshape((nfaces,-1))
        return self.eadj


    def nEdgeAdjacent(self):
        """Find the number of adjacent elems."""
        return (self.edgeAdjacency() >=0).sum(axis=-1)


    def nodeAdjacency(self):
        """Find the elems adjacent to elems via one or two nodes."""
        if self.adj is None:
            self.adj = adjacent(self.elems,self.nodeConnections())
        return self.adj


    def nNodeAdjacent(self):
        """Find the number of adjacent elems."""
        return (self.nodeAdjacency() >=0).sum(axis=-1)


    def surfaceType(self):
        ncon = self.nEdgeConnected()
        nadj = self.nEdgeAdjacent()
        maxcon = ncon.max()
        mincon = ncon.min()
        manifold = maxcon == 2
        closed = mincon == 2
        return manifold,closed,mincon,maxcon


    def borderEdges(self):
        """Detect the border elements of TriSurface.

        The border elements are the edges having less than 2 connected elements.
        Returns True where edge is on the border.
        """
        return self.nEdgeConnected() <= 1


    def borderEdgeNrs(self):
        """Returns the numbers of the border edges."""
        return where(self.nEdgeConnected() <= 1)[0]


    def borderNodeNrs(self):
        """Detect the border nodes of TriSurface.

        The border nodes are the vertices belonging to the border edges.
        Returns a list of vertex numbers.
        """
        border = self.getEdges()[self.borderEdgeNrs()]
        return unique1d(border)
        

    def isManifold(self):
        return self.surfaceType()[0] 


    def isClosedManifold(self):
        stype = self.surfaceType()
        return stype[0] and stype[1]


    def checkBorder(self):
        """Return the border of TriSurface as a set of segments."""
        border = self.getEdges()[self.borderEdges()]
        if len(border) > 0:
            return closedLoop(border)
        else:
            return None


    def fillBorder(self,method=0):
        """If the surface has a single closed border, fill it.

        Filling the border is done by adding a single point inside
        the border and connectin it with all border segments.
        This works well if the border is smooth and nearly planar.
        """
        border = self.getEdges()[self.borderEdges()]
        closed,loop = closedLoop(border)
        if closed == 0:
            if method == 0:
                coords,elems = surfaceInsideLoop(self.coords,loop)
                newS = TriSurface(coords,elems)
            else:
                elems = fillHole(self.coords,loop)
                newS = TriSurface(self.coords,elems)
            self.append(newS)


    def border(self):
        """Return the border of TriSurface as a Plex-2 Formex."""
        return Formex(self.coords[self.getEdges()[self.borderEdges()]])


    def edgeCosAngles(self):
        """Return the cos of the angles over all edges.
        
        The surface should be a manifold (max. 2 elements per edge).
        Edges with only one element get angles = 1.0.
        """
        conn = self.edgeConnections()
        # Bail out if some edge has more than two connected faces
        if conn.shape[1] != 2:
            raise RuntimeError,"TriSurface is not a manifold"
        angles = ones(self.nedges())
        conn2 = (conn >= 0).sum(axis=-1) == 2
        n = self.areaNormals()[1][conn[conn2]]
        angles[conn2] = dotpr(n[:,0],n[:,1])
        return angles


    def edgeAngles(self):
        """Return the angles over all edges (in degrees)."""
        return arccos(self.edgeCosAngles()) / Deg


    def data(self):
        """Compute data for all edges and faces."""
        if hasattr(self,'edglen'):
            return
        self.areaNormals()
        edg = self.coords[self.getEdges()]
        edglen = length(edg[:,1]-edg[:,0])
        facedg = edglen[self.getFaces()]
        edgmin = facedg.min(axis=-1)
        edgmax = facedg.max(axis=-1)
        altmin = 2*self.areas / edgmax
        aspect = edgmax/altmin
        self.edglen,self.facedg,self.edgmin,self.edgmax,self.altmin,self.aspect = edglen,facedg,edgmin,edgmax,altmin,aspect 


    def aspectRatio(self):
        self.data()
        return self.aspect

 
    def smallestAltitude(self):
        self.data()
        return self.altmin


    def longestEdge(self):
        self.data()
        return self.edgmax


    def shortestEdge(self):
        self.data()
        return self.edgmin

   
    def stats(self):
        """Return a text with full statistics."""
        bbox = self.bbox()
        manifold,closed,mincon,maxcon = self.surfaceType()
        self.data()
        angles = self.edgeAngles()
        area = self.area()
        if manifold and closed:
            volume = self.volume()
        else:
            volume = 0.0
        print(
            self.ncoords(),self.nedges(),self.nfaces(),
            bbox[0],bbox[1],
            mincon,maxcon,
            manifold,closed,
            self.areas.min(),self.areas.max(),
            self.edglen.min(),self.edglen.max(),
            self.altmin.min(),self.aspect.max(),
            angles.min(),angles.max(),
            area,volume
            )
        s = """
Size: %d vertices, %s edges and %d faces
Bounding box: min %s, max %s
Minimal/maximal number of connected faces per edge: %s/%s
Surface is manifold: %s; surface is closed: %s
Smallest area: %s; largest area: %s
Shortest edge: %s; longest edge: %s
Shortest altitude: %s; largest aspect ratio: %s
Angle between adjacent faces: smallest: %s; largest: %s
Total area: %s; Enclosed volume: %s   
""" % (
            self.ncoords(),self.nedges(),self.nfaces(),
            bbox[0],bbox[1],
            mincon,maxcon,
            manifold,closed,
            self.areas.min(),self.areas.max(),
            self.edglen.min(),self.edglen.max(),
            self.altmin.min(),self.aspect.max(),
            angles.min(),angles.max(),
            area,volume
            )
        return s


##################  Partitioning a surface #############################


    def edgeFront(self,startat=0,okedges=None,front_increment=1):
        """Generator function returning the frontal elements.

        startat is an element number or list of numbers of the starting front.
        On first call, this function returns the starting front.
        Each next() call returns the next front.
        front_increment determines how the property increases at each
        frontal step. There is an extra increment +1 at each start of
        a new part. Thus, the start of a new part can always be detected
        by a front not having the property of the previous plus front_increment.
        """
        #print("FRONT_INCREMENT %s" % front_increment)
        p = -ones((self.nfaces()),dtype=int)
        if self.nfaces() <= 0:
            return
        # Construct table of elements connected to each edge
        conn = self.edgeConnections()
        # Bail out if some edge has more than two connected faces
        if conn.shape[1] != 2:
            GD.warning("Surface is not a manifold")
            return
        # Check size of okedges
        if okedges is not None:
            if okedges.ndim != 1 or okedges.shape[0] != self.nedges():
                raise ValueError,"okedges has incorrect shape"

        # Remember edges left for processing
        todo = ones((self.nedges(),),dtype=bool)
        elems = clip(asarray(startat),0,self.nfaces())
        prop = 0
        while elems.size > 0:
            # Store prop value for current elems
            p[elems] = prop
            yield p

            prop += front_increment

            # Determine border
            edges = unique(self.getFaces()[elems])
            edges = edges[todo[edges]]
            if edges.size > 0:
                # flag edges as done
                todo[edges] = 0
                # take connected elements
                if okedges is None:
                    elems = conn[edges].ravel()
                else:
                    elems = conn[edges[okedges[edges]]].ravel()
                elems = elems[(elems >= 0) * (p[elems] < 0) ]
                if elems.size > 0:
                    continue

            # No more elements in this part: start a new one
            #print("NO MORE ELEMENTS")
            elems = where(p<0)[0]
            if elems.size > 0:
                # Start a new part
                elems = elems[[0]]
                prop += 1


    def nodeFront(self,startat=0,front_increment=1):
        """Generator function returning the frontal elements.

        startat is an element number or list of numbers of the starting front.
        On first call, this function returns the starting front.
        Each next() call returns the next front.
        """
        p = -ones((self.nfaces()),dtype=int)
        if self.nfaces() <= 0:
            return
        # Construct table of elements connected to each element
        adj = self.nodeAdjacency()

        # Remember nodes left for processing
        todo = ones((self.npoints(),),dtype=bool)
        elems = clip(asarray(startat),0,self.nfaces())
        prop = 0
        while elems.size > 0:
            # Store prop value for current elems
            p[elems] = prop
            yield p

            prop += front_increment

            # Determine adjacent elements
            elems = unique1d(adj[elems])
            elems = elems[(elems >= 0) * (p[elems] < 0) ]
            if elems.size > 0:
                continue

            # No more elements in this part: start a new one
            elems = where(p<0)[0]
            if elems.size > 0:
                # Start a new part
                elems = elems[[0]]
                prop += 1


    def walkEdgeFront(self,startat=0,nsteps=-1,okedges=None,front_increment=1):
        for p in self.edgeFront(startat=startat,okedges=okedges,front_increment=front_increment):
            if nsteps > 0:
                nsteps -= 1
            elif nsteps == 0:
                break
        return p


    def walkNodeFront(self,startat=0,nsteps=-1,front_increment=1):
        for p in self.nodeFront(startat=startat,front_increment=front_increment):
            if nsteps > 0:
                nsteps -= 1
            elif nsteps == 0:
                break
        return p


    def growSelection(self,sel,mode='node',nsteps=1):
        """Grows a selection of a surface.

        `p` is a single element number or a list of numbers.
        The return value is a list of element numbers obtained by
        growing the front `nsteps` times.
        The `mode` argument specifies how a single frontal step is done:

        - 'node' : include all elements that have a node in common,
        - 'edge' : include all elements that have an edge in common.
        """
        if mode == 'node':
            p = self.walkNodeFront(startat=sel,nsteps=nsteps)
        elif mode == 'edge':
            p = self.walkEdgeFront(startat=sel,nsteps=nsteps)
        return where(p>=0)[0]
    

    def partitionByEdgeFront(self,okedges,firstprop=0,startat=0):
        """Detects different parts of the surface using a frontal method.

        okedges flags the edges where the two adjacent triangles are to be
        in the same part of the surface.
        startat is a list of elements that are in the first part. 
        The partitioning is returned as a property type array having a value
        corresponding to the part number. The lowest property number will be
        firstprop
        """
        return firstprop + self.walkEdgeFront(startat=startat,okedges=okedges,front_increment=0)
    

    def partitionByNodeFront(self,firstprop=0,startat=0):
        """Detects different parts of the surface using a frontal method.

        okedges flags the edges where the two adjacent triangles are to be
        in the same part of the surface.
        startat is a list of elements that are in the first part. 
        The partitioning is returned as a property type array having a value
        corresponding to the part number. The lowest property number will be
        firstprop
        """
        return firstprop + self.walkNodeFront(startat=startat,front_increment=0)


    def partitionByConnection(self):
        return self.partitionByNodeFront()


    def partitionByAngle(self,angle=180.,firstprop=0,startat=0):
        conn = self.edgeConnections()
        # Flag edges that connect two faces
        conn2 = (conn >= 0).sum(axis=-1) == 2
        # compute normals and flag small angles over edges
        cosangle = cosd(angle)
        n = self.areaNormals()[1][conn[conn2]]
        small_angle = ones(conn2.shape,dtype=bool)
        small_angle[conn2] = dotpr(n[:,0],n[:,1]) >= cosangle
        return firstprop + self.partitionByEdgeFront(small_angle)


    def cutWithPlane(self,*args,**kargs):
        """Cut a surface with a plane."""
        self.__init__(self.toFormex().cutWithPlane(*args,**kargs))

    cutAtPlane = cutWithPlane  # DEPRECATED


    def connectedElements(self,target,elemlist=None):
        """Return the elements from list connected with target"""
        if elemlist is None:
            A = self
            elemlist = arange(self.nelems())
        else:
            A = self.select(elemlist)
        if target not in elemlist:
            return []
        
        p = A.partitionByConnection()
        prop = p[elemlist == target]
        return elemlist[p==prop]



    def intersectionWithPlane(self,p,n,atol=0.,ignoreErrors=False):
        """Return the intersection lines with plane (p,n).

        Returns a plex-2 mesh with the line segments obtained by cutting
        all triangles of the surface with the plane (p,n)
        p is a point specified by 3 coordinates.
        n is the normal vector to a plane, specified by 3 components.
        atol is a tolerance factor defining whether an edge is intersected
        by the plane.

        The return value is a plex-2 Mesh where the line segments defining
        the intersection are sorted to form continuous lines. The Mesh has
        property numbers such that all segments forming a single continuous
        part have the same property value.
        The splitProp() method can be used to get a list of Meshes.
        """
        # First, reduce the surface to the part interseting with the plane
        n = asarray(n)
        p = asarray(p)
        t = self.test(nodes='all',dir=n,min=p,atol=atol)
        u = self.test(nodes='all',dir=n,max=p,atol=atol)
        S = self.cclip(t+u)

        # If there is no intersection, we're done
        if S.nelems() == 0:
            return Mesh(Coords(),[])

        # Now, take the mesh with the edges, and intersect them
        edg = S.getEdges()
        fac = S.getFaces()
        M = Mesh(S.coords,edg)
        t = M.test(nodes='all',dir=n,min=p,atol=atol)
        u = M.test(nodes='all',dir=n,max=p,atol=atol)
        w = ((t+u) == 0) * (t*u == 0)
        ind = where(w)[0]
        rev = reverseUniqueIndex(ind)
        M = M.clip(w)
        x = M.toFormex().intersectionPointsWithPlane(p,n).coords.reshape(-1,3)

        Mparts = []
        
        # edges returning NaN as intersection are inside the cutting plane
        inside = ind[isnan(x).any(axis=1)]
        if len(inside) > 0:
            # Mark these edges as non-cutting, to avoid other triangles
            # to pick them up again
            w[inside] = False

            # Keep these edges in the return value
            # BUT ONLY if they occur only once: NEEDS TO BE IMPLEMENTED
            edgcon = S.edgeConnections()[inside]
            alledg = fac[edgcon].reshape(-1,6)
            keep = array([ setdiff1d(two,inside).size  for two in alledg ])
            cutins = edg[inside[keep>0]]
            Mparts.append(Mesh(M.coords,cutins).compact())

        # Split the triangles based on the number of cutting edges
        # 0 : should not occur: filtered at beginning
        # 1 : currently ignored: does not generate a line segment
        # 2 : the normal case
        # 3 : either the triangle is completely in the plane (handled above)
        #     or two cutting points will coincide with the same vertex.
        #     The latter is currently unhandled: an error is raised.
        cut = w[fac]
        ncut = cut.sum(axis=1)
        icut = [ where(ncut==i)[0] for i in range(4) ]
        cut0,cut1,cut2,cut3 = icut
        GD.debug("Number of triangles with 0..3 cutting edges: %s" % icut)
        
        if cut3.size > 0:
            # The triangles with three vertices in the cutting plane
            # have already been handled above, so these must be cases
            # with one edge and the opposite vertex cutting the plane.
            # Our strategy is to take the 3 cutting points (two of whom
            # are coincident) and to fuse them. The fusing information
            # then tells us which are the noncoincident vertices that
            # define the intersection line segment.
            cutedg = fac[cut3][cut[cut3]].reshape(-1,3)
            seg = rev[cutedg]
            xd,xe = x[seg].fuse()
            # Keep the first vertex and the 2nd or 3rd, depending on which
            # is different from the first
            seg = column_stack([xe[:,0], where(xe[:,1] == xe[:,0],xe[:,2],xe[:,1])])
            Mparts.append(Mesh(xd,seg))


        if cut2.size > 0:
            # Create line elements between each pair of intersection points
            cutedg = fac[cut2][cut[cut2]].reshape(-1,2)
            seg = rev[cutedg]
            Mparts.append(Mesh(x,seg).compact())

        # Done with getting the segments
        if len(Mparts) ==  0:
            # No intersection: return empty mesh
            return Mesh(Coords(),[])

        if len(Mparts) == 1:
            M = Mparts[0]
        else:
            M = Mesh.concatenate(Mparts)

        # Remove degenerate and doubles
        M = Mesh(M.coords,M.elems.removeDegenerate().removeDoubles())
            
        # Split in connected loops
        parts = connectedLineElems(M.elems)
        prop = concatenate([ [i]*p.nelems() for i,p in enumerate(parts)])
        elems = concatenate(parts,axis=0)

        return Mesh(M.coords,elems,prop=prop)


    def slice(self,dir=0,nplanes=20,ignoreErrors=False):
        """Intersect a surface with a sequence of planes.

        A sequence of nplanes planes with normal dir is constructed
        at equal distances spread over the bbox of the surface.

        The return value is a list of intersectionWithPlanes() return
        values, i.e. a list of list of meshes.
        """
        o = self.center()
        xmin,xmax = self.coords.directionalExtremes(dir,o)
        P = Coords.interpolate(xmin,xmax,nplanes)
        return [ self.intersectionWithPlane(p,dir,ignoreErrors=ignoreErrors) for p in P ]


##################  Smooth a surface #############################

    def smoothLowPass(self,n_iterations=2,lambda_value=0.5):
        """Smooth the surface using a low-pass filter."""
        k = 0.1
        mu_value = -lambda_value/(1-k*lambda_value)
        # find adjacency
        adj = adjacencyArray(self.getEdges())
        # find interior vertices
        bound_edges = self.borderEdgeNrs()
        inter_vertex = resize(True,self.ncoords())
        inter_vertex[unique1d(self.getEdges()[bound_edges])] = False
        # calculate weights
        w = ones(adj.shape,dtype=float)
        w[adj<0] = 0.
        val = (adj>=0).sum(-1).reshape(-1,1)
        w /= val
        w = w.reshape(adj.shape[0],adj.shape[1],1)
        # recalculate vertices
        p = self.coords
        for step in range(n_iterations/2):
            p[inter_vertex] = p[inter_vertex] + lambda_value*(w[inter_vertex]*(p[adj[inter_vertex]]-p[inter_vertex].reshape(-1,1,3))).sum(1)
            p[inter_vertex] = p[inter_vertex] + mu_value*(w[inter_vertex]*(p[adj[inter_vertex]]-p[inter_vertex].reshape(-1,1,3))).sum(1)


    def smoothLaplaceHC(self,n_iterations=2,lambda_value=0.5,alpha=0.,beta=0.2):
        """Smooth the surface using a Laplace filter and HC algorithm."""
        # find adjacency
        adj = adjacencyArray(self.getEdges())        
        # find interior vertices
        bound_edges = self.borderEdgeNrs()
        inter_vertex = resize(True,self.ncoords())
        inter_vertex[unique1d(self.getEdges()[bound_edges])] = False
        # calculate weights
        w = ones(adj.shape,dtype=float)
        w[adj<0] = 0.
        val = (adj>=0).sum(-1).reshape(-1,1)
        w /= val
        w = w.reshape(adj.shape[0],adj.shape[1],1)
        # recalculate vertices
        o = self.coords.copy()
        p = self.coords
        for step in range(n_iterations):
            pn = p + lambda_value*(w*(p[adj]-p.reshape(-1,1,3))).sum(1)
            b = pn - (alpha*o + (1-alpha)*p)
            p[inter_vertex] = pn[inter_vertex] - (beta*b[inter_vertex] + (1-beta)*(w[inter_vertex]*b[adj[inter_vertex]]).sum(1))


########################## Methods using GTS #############################

    def check(self,verbose=False):
        """Check the surface using gtscheck."""
        cmd = 'gtscheck'
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Checking with command\n %s" % cmd)
        cmd += ' < %s' % tmp
        sta,out = runCommand(cmd,False)
        os.remove(tmp)
        GD.message(out)
        if sta == 0:
            GD.message('The surface is a closed, orientable non self-intersecting manifold')
 

    def split(self,base,verbose=False):
        """Check the surface using gtscheck."""
        cmd = 'gtssplit -v %s' % base
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Splitting with command\n %s" % cmd)
        cmd += ' < %s' % tmp
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            GD.message(out)
   

    def coarsen(self,min_edges=None,max_cost=None,
                mid_vertex=False, length_cost=False, max_fold = 1.0,
                volume_weight=0.5, boundary_weight=0.5, shape_weight=0.0,
                progressive=False, log=False, verbose=False):
        """Coarsen the surface using gtscoarsen."""
        if min_edges is None and max_cost is None:
            min_edges = self.nedges() / 2
        cmd = 'gtscoarsen'
        if min_edges:
            cmd += ' -n %d' % min_edges
        if max_cost:
            cmd += ' -c %f' % max_cost
        if mid_vertex:
            cmd += ' -m'
        if length_cost:
            cmd += ' -l'
        if max_fold:
            cmd += ' -f %f' % max_fold
        cmd += ' -w %f' % volume_weight
        cmd += ' -b %f' % boundary_weight
        cmd += ' -s %f' % shape_weight
        if progressive:
            cmd += ' -p'
        if log:
            cmd += ' -L'
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Coarsening with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            GD.message(out)
        GD.message("Reading coarsened model from %s" % tmp1)
        self.__init__(*read_gts(tmp1))        
        os.remove(tmp1)
   

    def refine(self,max_edges=None,min_cost=None,
               log=False, verbose=False):
        """Refine the surface using gtsrefine."""
        if max_edges is None and min_cost is None:
            max_edges = self.nedges() * 2
        cmd = 'gtsrefine'
        if max_edges:
            cmd += ' -n %d' % max_edges
        if min_cost:
            cmd += ' -c %f' % min_cost
        if log:
            cmd += ' -L'
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Refining with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            GD.message(out)
        GD.message("Reading refined model from %s" % tmp1)
        self.__init__(*read_gts(tmp1))        
        os.remove(tmp1)


    def smooth(self,lambda_value=0.5,n_iterations=2,
               fold_smoothing=None,verbose=False):
        """Smooth the surface using gtssmooth."""
        cmd = 'gtssmooth'
        if fold_smoothing:
            cmd += ' -f %s' % fold_smoothing
        cmd += ' %s %s' % (lambda_value,n_iterations)
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Smoothing with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            GD.message(out)
        GD.message("Reading smoothed model from %s" % tmp1)
        self.__init__(*read_gts(tmp1))        
        os.remove(tmp1)


#### THIS FUNCTION RETURNS A NEW SURFACE
#### WE MIGHT DO THIS IN FUTURE FOR ALL SURFACE PROCESSING

    def boolean(self,surf,op,inter=False,check=False,verbose=False):
        """Perform a boolean operation with surface surf.

        """
        ops = {'+':'union', '-':'diff', '*':'inter'}
        cmd = 'gtsset'
        if inter:
            cmd += ' -i'
        if check:
            cmd += ' -s'
        if verbose:
            cmd += ' -v'
        cmd += ' '+ops[op]
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        tmp2 = tempfile.mktemp('.stl')
        GD.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        GD.message("Writing temp file %s" % tmp1)
        surf.write(tmp1,'gts')
        GD.message("Performing boolean operation with command\n %s" % cmd)
        cmd += ' %s %s | gts2stl > %s' % (tmp,tmp1,tmp2)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        os.remove(tmp1)
        if sta or verbose:
            GD.message(out)
        GD.message("Reading result from %s" % tmp2)
        S = TriSurface.read(tmp2)        
        os.remove(tmp2)
        return S


##########################################################################
################# Non-member and obsolete functions ######################

def read_error(cnt,line):
    """Raise an error on reading the stl file."""
    raise RuntimeError,"Invalid .stl format while reading line %s\n%s" % (cnt,line)


def degenerate(area,norm):
    """Return a list of the degenerate faces according to area and normals.

    A face is degenerate if its surface is less or equal to zero or the
    normal has a nan.
    """
    return unique(concatenate([where(area<=0)[0],where(isnan(norm))[0]]))


def read_stla(fn,dtype=Float,large=False,guess=True):
    """Read an ascii .stl file into an [n,3,3] float array.

    If the .stl is large, read_ascii_large() is recommended, as it is
    a lot faster.
    """
    if large:
        return read_ascii_large(fn,dtype=dtype)
    if guess:
        n = countLines(fn) / 7 # ASCII STL has 7 lines per triangle
    else:
        n = 100
    f = file(fn,'r')
    a = zeros(shape=[n,3,3],dtype=dtype)
    x = zeros(shape=[3,3],dtype=dtype)
    i = 0
    j = 0
    cnt = 0
    finished = False
    for line in f:
        cnt += 1
        s = line.strip().split()
        if s[0] == 'vertex':
            if j >= 3:
                read_error(cnt,line)
            x[j] = map(float,s[1:4])
            j += 1
        elif s[0] == 'outer':
            j = 0
        elif s[0] == 'endloop':
            a[i] = x
        elif s[0] == 'facet':
            if i >= a.shape[0]:
                # increase the array size
                a.resize([2*a.shape[0],3,3])
        elif s[0] == 'endfacet':
            i += 1
        elif s[0] == 'solid':
            pass
        elif s[0] == 'endsolid':
            finished = True
            break
    if f:    
        f.close()
    if finished:
        return a[:i]
    raise RuntimeError,"Incorrect stl file: read %d lines, %d facets" % (cnt,i)
        

def read_ascii_large(fn,dtype=Float):
    """Read an ascii .stl file into an [n,3,3] float array.

    This is an alternative for read_ascii, which is a lot faster on large
    STL models.
    It requires the 'awk' command though, so is probably only useful on
    Linux/UNIX. It works by first transforming  the input file to a
    .nodes file and then reading it through numpy's fromfile() function.
    """
    tmp = '%s.nodes' % fn
    runCommand("awk '/^[ ]*vertex[ ]+/{print $2,$3,$4}' '%s' | d2u > '%s'" % (fn,tmp))
    nodes = fromfile(tmp,sep=' ',dtype=dtype).reshape((-1,3,3))
    return nodes


def off_to_tet(fn):
    """Transform an .off model to tetgen (.node/.smesh) format."""
    GD.message("Transforming .OFF model %s to tetgen .smesh" % fn)
    nodes,elems = read_off(fn)
    write_node_smesh(changeExt(fn,'.smesh'),nodes,elems)


def find_row(mat,row,nmatch=None):
    """Find all rows in matrix matching given row."""
    if nmatch is None:
        nmatch = mat.shape[1]
    return where((mat == row).sum(axis=1) == nmatch)[0]


def find_nodes(nodes,coords):
    """Find nodes with given coordinates in a node set.

    nodes is a (nnodes,3) float array of coordinates.
    coords is a (npts,3) float array of coordinates.

    Returns a (n,) integer array with ALL the node numbers matching EXACTLY
    ALL the coordinates of ANY of the given points.
    """
    return concatenate([ find_row(nodes,c) for c in coords])


def find_first_nodes(nodes,coords):
    """Find nodes with given coordinates in a node set.

    nodes is a (nnodes,3) float array of coordinates.
    coords is a (npts,3) float array of coordinates.

    Returns a (n,) integer array with THE FIRST node number matching EXACTLY
    ALL the coordinates of EACH of the given points.
    """
    res = [ find_row(nodes,c) for c in coords ]
    return array([ r[0] for r in res ])


def find_triangles(elems,triangles):
    """Find triangles with given node numbers in a surface mesh.

    elems is a (nelems,3) integer array of triangles.
    triangles is a (ntri,3) integer array of triangles to find.
    
    Returns a (ntri,) integer array with the triangles numbers.
    """
    magic = elems.max()+1

    mag1 = enmagic3(elems,magic)
    mag2 = enmagic3(triangles,magic)

    nelems = elems.shape[0]
    srt = mag1.argsort()
    old = arange(nelems)[srt]
    mag1 = mag1[srt]
    pos = mag1.searchsorted(mag2)
    tri = where(mag1[pos]==mag2, old[pos], -1)
    return tri
    

def remove_triangles(elems,remove):
    """Remove triangles from a surface mesh.

    elems is a (nelems,3) integer array of triangles.
    remove is a (nremove,3) integer array of triangles to remove.
    
    Returns a (nelems-nremove,3) integer array with the triangles of
    nelems where the triangles of remove have been removed.
    """
    GD.message("Removing %s out of %s triangles" % (remove.shape[0],elems.shape[0]))
    magic = elems.max()+1

    mag1 = enmagic3(elems,magic)
    mag2 = enmagic3(remove,magic)

    mag1.sort()
    mag2.sort()

    nelems = mag1.shape[0]

    pos = mag1.searchsorted(mag2)
    mag1[pos] = -1
    mag1 = mag1[mag1 >= 0]

    elems = demagic3(mag1,magic)
    GD.message("Actually removed %s triangles, leaving %s" % (nelems-mag1.shape[0],elems.shape[0]))

    return elems


### Some simple surfaces ###

def Rectangle(nx,ny):
    """Create a plane rectangular surface consisting of a nx,ny grid."""
    F = Formex(mpattern('12-34')).replic2(nx,ny,1,1)    
    return TriSurface(F)

def Cube():
    """Create a surface in the form of a cube"""
    back = Formex(mpattern('12-34'))
    fb = back.reverse() + back.translate(2,1)
    faces = fb + fb.rollAxes(1) + fb.rollAxes(2)
    return TriSurface(faces)

def Sphere(level=4,verbose=False,filename=None):
    """Create a spherical surface by caling the gtssphere command.

    If a filename is given, it is stored under that name, else a temporary
    file is created.
    Beware: this may take a lot of time if level is 8 or higher.
    """
    cmd = 'gtssphere '
    if verbose:
        cmd += ' -v'
    cmd += ' %s' % level
    if filename is None:
        tmp = tempfile.mktemp('.gts')
    else:
        tmp = filename
    cmd += ' > %s' % tmp
    GD.message("Writing file %s" % tmp)
    sta,out = runCommand(cmd)
    if sta or verbose:
        GD.message(out)
    GD.message("Reading model from %s" % tmp)
    S = TriSurface.read(tmp)
    if filename is None:
        os.remove(tmp)
    return S


# For compatibility

#Surface = TriSurface

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print(a)
    
# End
