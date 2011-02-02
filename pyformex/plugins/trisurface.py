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
"""Operations on triangulated surfaces.

A triangulated surface is a surface consisting solely of triangles.
Any surface in space, no matter how complex, can be approximated with
a triangulated surface.
"""

import pyformex as pf

import filewrite as filewrite

from formex import *
from plugins import tetgen
from mesh import Mesh
from connectivity import Connectivity,connectedLineElems,adjacencyArrays
from utils import runCommand, changeExt,countLines,mtime,hasExternal
from gui.drawable import interpolateNormals
from plugins.geomtools import projectionVOP,rotationAngle,facetDistance,edgeDistance,vertexDistance, triangleBoundingCircle
from plugins import inertia

import os,tempfile

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
        outname = pf.cfg.get('surface/stlread','.off')
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
    pf.message("Reading GTS file %s" % fn)
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
    pf.message("Read %d coords, %d edges, %d faces" % (ncoords,nedges,nfaces))
    if coords.shape[0] != ncoords or \
       edges.shape[0] != nedges or \
       faces.shape[0] != nfaces:
        pf.message("Error while reading GTS file: the file is probably incorrect!")
    return coords,edges,faces


def read_off(fn):
    """Read an OFF surface mesh.

    The mesh should consist of only triangles!
    Returns a nodes,elems tuple.
    """
    pf.message("Reading .OFF %s" % fn)
    fil = file(fn,'r')
    head = fil.readline().strip()
    if head != "OFF":
        print("%s is not an OFF file!" % fn)
        return None,None
    nnodes,nelems,nedges = map(int,fil.readline().split())
    nodes = fromfile(file=fil, dtype=Float, count=3*nnodes, sep=' ')
    # elems have number of vertices + 3 vertex numbers
    elems = fromfile(file=fil, dtype=int32, count=4*nelems, sep=' ')
    pf.message("Read %d nodes and %d elems" % (nnodes,nelems))
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
        pf.debug("Error during conversion of file '%s' to '%s'" % (fn,ofn))
        pf.debug(out)
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
    scr = os.path.join(pf.cfg['bindir'],'gambit-neu ')
    runCommand("%s '%s'" % (scr,fn))
    nodesf = changeExt(fn,'.nodes')
    elemsf = changeExt(fn,'.elems')
    nodes = fromfile(nodesf,sep=' ',dtype=Float).reshape((-1,3))
    elems = fromfile(elemsf,sep=' ',dtype=int32).reshape((-1,3))
    return nodes, elems-1


# Output of surface file formats

def write_stla(f,x):
    """Export an x[n,3,3] float array as an ascii .stl file."""

    own = type(f) == str
    if own:
        f = file(f,'w')
    f.write("solid  Created by %s\n" % pf.Version)
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
    """Calculate curvature parameters at the nodes
    
    (according to Dong and Wang 2005;
    Koenderink and Van Doorn 1992).
    This uses the nodes that are connected to the node via a shortest
    path of 'neighbours' edges.
    Eight values are returned: the Gaussian and mean curvature, the
    shape index, the curvedness, the principal curvatures and the
    principal directions.
    """
    # calculate n-ring neighbourhood of the nodes (n=neighbours)
    adj = adjacencyArrays(edges,nsteps=neighbours)[-1]
    adjNotOk = adj<0
    # for nodes that have less than three adjacent nodes, remove the adjacencies
    adjNotOk[(adj>=0).sum(-1) <= 2] = True
    # calculate unit length average normals at the nodes p
    # a weight 1/|gi-p| could be used (gi=center of the face fi)
    p = coords
    n = interpolateNormals(coords,elems,atNodes=True)
    # double-precision: this will allow us to check the sign of the angles    
    p = p.astype(float64)
    n = n.astype(float64)
    vp = p[adj] - p[:,newaxis]
    vn = n[adj] - n[:,newaxis]    
    # where adjNotOk, set vectors = [0.,0.,0.]
    # this will result in NaN values
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
    tmax1 = tmax
    tmax2 = cross(n,tmax1)
    tmax2 = normalize(tmax2)
    # calculate angles (tmax1,t)
    theta,rot = rotationAngle(repeat(tmax1[:,newaxis],t.shape[1],1),t,angle_spec=Rad)
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
    # calculate the Gaussian and mean curvature
    K = a*c-b**2/4
    H = (a+c)/2
    # calculate the principal curvatures and principal directions
    k1 = H+sqrt(H**2-K)
    k2 = H-sqrt(H**2-K)
    theta0 = 0.5*arcsin(b/(k2-k1))
    w = apply_along_axis(isClose,0,-b,2*(k2-k1)*cos(theta0)*sin(theta0))
    theta0[w] = pi-theta0[w]
    e1 = cos(theta0)[:,newaxis]*tmax1+sin(theta0)[:,newaxis]*tmax2
    e2 = cos(theta0)[:,newaxis]*tmax2-sin(theta0)[:,newaxis]*tmax1
    # calculate the shape index and curvedness
    S = 2./pi*arctan((k1+k2)/(k1-k2))
    C = square((k1**2+k2**2)/2)
    return K,H,S,C,k1,k2,e1,e2


############################################################################


def surfaceInsideBorder(border,method='radial'):
    """Create a surface inside a closed curve defined by a 2-plex Mesh.

    border is a 2-plex Mesh representing a closed polyline.

    The return value is a TrisSurface filling the hole inside the border.

    There are two fill methods:
    
    - 'radial': this method adds a central point and connects all border
      segments with the center to create triangles. It is fast and works
      well if the border is smooth, nearly convex and nearly planar.
    - 'border': this method creates subsequent triangles by connecting the
      endpoints of two consecutive border segments and thus works its way
      inwards until the hole is closed. Triangles are created at the segments
      that form the smallest angle. This method is slower, but works also
      for most complex borders. Also, because it does not create any new
      points, the returned surface uses the same point coordinate array
      as the input Mesh.
    """
    if border.nplex() != 2:
        raise ValueError,"Expected Mesh with plexitude 2, got %s" % border.nplex()

    if method == 'radial':
        x = border.getPoints().center()
        n = zeros_like(border.elems[:,:1]) + border.coords.shape[0]
        elems = concatenate([border.elems,n],axis=1)
        coords = Coords.concatenate([border.coords,x])

    elif method == 'border':
        coords = border.coords
        segments = border.elems
        elems = empty((0,3,),dtype=int)
        while len(segments) != 3:
            segments,triangle = _create_border_triangle(coords,segments)
            elems = row_stack([elems,triangle])
        elems = row_stack([elems,segments[:,0]])

    else:
        raise ValueError,"Strategy should be either 'radial' or 'border'"
    
    return TriSurface(coords,elems)


# This should be replaced with an algorith using PolyLine
def _create_border_triangle(coords,elems):
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
        self.areas = self.normals = None
        self.adj = None
        if hasattr(self,'edglen'):
            del self.edglen
            
        if len(args) == 0:
            return  # an empty surface
        
        if len(args) == 1:
            # argument should be a suitably structured geometry object
            # TriSurface, Mesh, Formex, Coords, ndarray, ...
            a = args[0]

            if isinstance(a,Mesh):
                if a.nplex() != 3 or a.eltype.name() != 'tri3':
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

                elems = faces.combine(edges)
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
        return self.getElemEdges().shape[0]

    def vertices(self):
        return self.coords
    
    def shape(self):
        """Return the number of points, edges, faces of the TriSurface."""
        return self.ncoords(),self.nedges(),self.nfaces()

    
    def getElemEdges(self):
        """Get the faces' edge numbers."""
        if self.face_edges is None:
            self.face_edges,self.edges = self.elems.insertLevel([[0,1],[1,2],[2,0]])
        return self.face_edges


###########################################################################
    #
    #   Operations that change the TriSurface itself
    #
    #  Make sure that you know what you're doing if you use these
    #
    #
    # Changes to the geometry should by preference be done through the
    # __init__ function, to ensure consistency of the data.
    # Convenience functions are defined to change some of the data.
    #

    def setCoords(self,coords):
        """Change the coords."""
        self.__init__(coords,self.elems,prop=self.prop)
        return self

    def setElems(self,elems):
        """Change the elems."""
        self.__init__(self.coords,elems,prop=self.prop)

    def setEdgesAndFaces(self,edges,faces):
        """Change the edges and faces."""
        self.__init__(self.coords,edges,faces,prop=self.prop)


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

   


###########################################################################
    #
    #   read and write
    #

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
            ftype = 'off'
        else:
            ftype = ftype.strip('.').lower()

        pf.message("Writing surface to file %s" % fname)
        if ftype == 'pgf':
            Geometry.write(self,fname)
        elif ftype == 'gts':
            filewrite.writeGTS(fname,self.coords,self.getEdges(),self.getElemEdges())
            pf.message("Wrote %s vertices, %s edges, %s faces" % self.shape())
        elif ftype in ['stl','off','smesh']:
            if ftype == 'stl':
                write_stla(fname,self.coords[self.elems])
            elif ftype == 'off':
                filewrite.writeOFF(fname,self.coords,self.elems)
            elif ftype == 'smesh':
                write_smesh(fname,self.coords,self.elems)
            pf.message("Wrote %s vertices, %s elems" % (self.ncoords(),self.nelems()))
        else:
            print("Cannot save TriSurface as file %s" % fname)

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

        This uses the nodes that are connected to the node via a shortest
        path of 'neighbours' edges.
        Eight values are returned: the Gaussian and mean curvature, the
        shape index, the curvedness, the principal curvatures and the
        principal directions.
        """
        curv = curvature(self.coords,self.elems,self.getEdges(),neighbours=neighbours)
        return curv
    
    
    def inertia(self):
        """Return inertia related quantities of the surface.
        
        This returns the center of gravity, the principal axes of inertia,
        the principal moments of inertia and the inertia tensor.
        """
        ctr,I = inertia.inertia(self.centroids(),mass=self.facetArea().reshape(-1,1))
        Iprin,Iaxes = inertia.principal(I,sort=True,right_handed=True)
        data = (ctr,Iaxes,Iprin,I)
        return data


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
        return unique(border)
        

    def isManifold(self):
        return self.surfaceType()[0] 


    def isClosedManifold(self):
        stype = self.surfaceType()
        return stype[0] and stype[1]


    def checkBorder(self):
        """Return the border of TriSurface as a set of segments."""
        border = self.getEdges()[self.borderEdges()]
        if len(border) > 0:
            return connectedLineElems(border)
        else:
            return []
        

    def border(self,):
        """Return the border(s) of TriSurface.

        The complete border of the surface is returned as a list
        of plex-2 Meshes. Each Mesh constitutes a continuous part
        of the border.
        """
        return [ Mesh(self.coords,e) for e in self.checkBorder() ]


    def fillBorder(self,method='radial',merge=False):
        """Fill the border areas of a surface to make it closed.

        If the surface has no border, it is returned unchanged.
        Else, each singly connected part of the border which forms a
        closed curve will be filled with triangles, thus effectively
        creating a closed surface.

        There are two methods, corresponding with the methods of
        the surfaceInsideBorder.
        """
        return [ surfaceInsideBorder(b,method).setProp(i) for i,b in enumerate(self.border()) ]


    # BV: There are too many of these! cfr getBorder,
    def boundaryEdges(self):
        """_Returns the border edges of a surface, grouped by id property. """
        be=Mesh(self.coords, self.getEdges()[self.borderEdges()] )#.renumber()
        be.elems=be.elems.removeDegenerate().removeDoubles()
        parts = connectedLineElems(be.elems)
        prop = concatenate([ [i]*p.nelems() for i,p in enumerate(parts)])
        elems = concatenate(parts,axis=0)
        return Mesh(be.coords,elems,prop=prop)
    
    # BV: There are too many of these!
    def boundaryFiller(self):
        """_Fills the holes of a surface by creating extra faces at the boundary edges. Original surface and boundaries are returned with different id prop."""
        brd=self.boundaryEdges()
        Brd=[ brd.withProp(p).compact() for p in brd.propSet() ]
        if self.prop==None:self=self.setProp(0)
        maxp=self.maxProp()+1
        capsm=[Mesh( surfaceInsideLoop(Brd[i].coords,Brd[i].elems) ).setProp(i+maxp) for i in range(len(Brd))]
        return TriSurface( Mesh.concatenate( capsm+[self] ) )#.renumber()


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
        return angles.clip(min=-1.,max=1.)


    def edgeAngles(self):
        """Return the angles over all edges (in degrees). It is the angle (0 to 180) between 2 face normals."""
        return arccos(self.edgeCosAngles()) / Deg


    def _compute_data(self):
        """Compute data for all edges and faces."""
        if hasattr(self,'edglen'):
            return
        self.areaNormals()
        edg = self.coords[self.getEdges()]
        edglen = length(edg[:,1]-edg[:,0])
        facedg = edglen[self.getElemEdges()]
        edgmin = facedg.min(axis=-1)
        edgmax = facedg.max(axis=-1)
        altmin = 2*self.areas / edgmax
        aspect = edgmax/altmin
        self.edglen,self.facedg,self.edgmin,self.edgmax,self.altmin,self.aspect = edglen,facedg,edgmin,edgmax,altmin,aspect 


    def aspectRatio(self):
        self._compute_data()
        return self.aspect

 
    def smallestAltitude(self):
        self._compute_data()
        return self.altmin


    def longestEdge(self):
        self._compute_data()
        return self.edgmax


    def shortestEdge(self):
        self._compute_data()
        return self.edgmin

   
    def stats(self):
        """Return a text with full statistics."""
        bbox = self.bbox()
        manifold,closed,mincon,maxcon = self.surfaceType()
        self._compute_data()
        angles = self.edgeAngles()
        vangles = self.getAngles()
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
            vangles.min(),vangles.max(),
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
Angle at vertices: smallest: %s; largest: %s
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
            vangles.min(),vangles.max(),
            area,volume
            )
        return s

    
    def distanceOfPoints(self,X,return_points=False):
        """Find the distances of points X to the TriSurface.
    
        The distance of a point is either:
        - the closest perpendicular distance to the facets;
        - the closest perpendicular distance to the edges;
        - the closest distance to the vertices.
    
        X is a (nX,3) shaped array of points.
        If return_points = True, a second value is returned: an array with
        the closest (foot)points matching X.
        """
        # distance from vertices
        Vp = self.coords
        res = vertexDistance(X,Vp,return_points) # OKdist, (OKpoints)
        dist = res[0]
        if return_points:
            points = res[1]

        # distance from edges
        Ep = self.coords[self.getEdges()]
        res = edgeDistance(X,Ep,return_points) # OKpid, OKdist, (OKpoints)
        okE,distE = res[:2]
        closer = distE < dist[okE]
        dist[okE[closer]] = distE[closer]
        if return_points:
            points[okE[closer]] = res[2][closer]

        # distance from faces
        Fp = self.coords[self.elems]
        res = facetDistance(X,Fp,return_points) # OKpid, OKdist, (OKpoints)
        okF,distF = res[:2]
        closer = distF < dist[okF]
        dist[okF[closer]] = distF[closer]
        if return_points:
            points[okF[closer]] = res[2][closer]

        if return_points:
            return dist,points
        else:
            return dist


##################  Transform surface #############################
    # All transformations now return a new surface

    def offset(self,distance=1.):
        """Offset a surface with a certain distance.
        
        All the nodes of the surface are translated over a specified distance
        along their normal vector.
        """
        n = self.avgVertexNormals()
        coordsNew = self.coords + NPA*distance
        return TriSurface(coordsNew,self.getElems(),prop=self.prop)


    def reflect(self,*args,**kargs):
        """Reflect the Surface in direction dir against plane at pos.

        Parameters:

        - `dir`: int: direction of the reflection (default 0)
        - `pos`: float: offset of the mirror plane from origin (default 0.0)
        - `inplace`: boolean: change the coordinates inplace (default False)
        - `reverse`: boolean: revert the normals of the triangles
          (default True).
          Reflection of the coordinates of a 2D Mesh reverses the surface
          sides. Setting this parameter True will cause an extra
          reversion. This is what is expected in most surface mirroring
          operations.
        """
        return Mesh.reflect(self,*args,**kargs)
    

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
            pf.warning("Surface is not a manifold")
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
            edges = unique(self.getElemEdges()[elems])
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
            elems = unique(adj[elems])
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
        """Grow a selection of a surface.

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
        """Detect different parts of the surface using a frontal method.

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
        firstprop.
        """
        return firstprop + self.walkNodeFront(startat=startat,front_increment=0)


    def partitionByConnection(self):
        """Detect the connected parts of a surface.

        The surface is partitioned in parts in which all elements are
        connected. Two elements are connected if it is possible to draw a
        continuous (poly)line from a point in one element to a point in
        the other element without leaving the surface.
        
        The partitioning is returned as a property type array having a value
        corresponding to the part number. The lowest property number will be
        firstprop.
        """
        return self.partitionByNodeFront()


    def partitionByAngle(self,angle=60.,firstprop=0,startat=0):
        """Partition the surface by splitting it at sharp edges.

        The surface is partitioned in parts in which all elements can be
        reach without ever crossing a sharp edge angle. More precisely,
        any two elements that can be connected by a line not crossing an
        edge between two elements having their normals differ more than
        angle (in degrees), will belong to the same part.
       
        The partitioning is returned as a property type array having a value
        corresponding to the part number. The lowest property number will be
        firstprop.
        """
        conn = self.edgeConnections()
        # Flag edges that connect two faces
        conn2 = (conn >= 0).sum(axis=-1) == 2
        # compute normals and flag small angles over edges
        cosangle = cosd(angle)
        n = self.areaNormals()[1][conn[conn2]]
        small_angle = ones(conn2.shape,dtype=bool)
        small_angle[conn2] = dotpr(n[:,0],n[:,1]) >= cosangle
        return firstprop + self.partitionByEdgeFront(small_angle)


    def splitByConnection(self):
        """Split the surface into connected parts.

        Returns a list of surfaces that each form a connected part.
        """
        split = self.splitProp(self.partitionByConnection())
        if split:
            return split.values()
        else:
            return [ self ]


    def largestByConnection(self):
        """Return the largest connected part of the surface."""
        p = self.partitionByConnection()
        nparts = p.max()+1
        if nparts == 1:
            return self,nparts
        else:
            t = [ p == pi for pi in range(nparts) ]
            n = [ ti.sum() for ti in t ]
            w = array(n).argmax()
            return self.clip(t[w]),nparts


    def cutWithPlane(self,*args,**kargs):
        """Cut a surface with a plane or a set of planes.

        Cuts the surface with one or more plane and returns either one side
        or both.

        Parameters:
        
        - `p`,`n`: a point and normal vector defining the cutting plane.
          p and n can be sequences of points and vector,
          allowing to cut with multiple planes.
          Both p and n have shape (3) or (npoints,3).

        The parameters are the same as in :meth:`Formex.CutWithPlane`.
        The returned surface will have its normals fixed wherever possible.
        """
        return TriSurface(self.toFormex().cutWithPlane(*args,**kargs)).fixNormals()


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


    def intersectionWithPlane(self,p,n):
        """Return the intersection lines with plane (p,n).

        Returns a plex-2 mesh with the line segments obtained by cutting
        all triangles of the surface with the plane (p,n)
        p is a point specified by 3 coordinates.
        n is the normal vector to a plane, specified by 3 components.

        The return value is a plex-2 Mesh where the line segments defining
        the intersection are sorted to form continuous lines. The Mesh has
        property numbers such that all segments forming a single continuous
        part have the same property value.
        The splitProp() method can be used to get a list of Meshes.
        """
        n = asarray(n)
        p = asarray(p)
        # The vertices are classified based on their distance d from the plane:
        # - inside: d = 0
        # - up: d > 0
        # - down: d < 0
        
        # First, reduce the surface to the part intersecting with the plane:
        # remove triangles with all up or all down vertices
        d = self.distanceFromPlane(p,n)
        d_ele = d[self.elems]
        ele_all_up = (d_ele > 0.).all(axis=1)
        ele_all_do = (d_ele < 0.).all(axis=1)
        S = self.cclip(ele_all_up+ele_all_do)

        # If there is no intersection, we're done
        if S.nelems() == 0:
            return Mesh(Coords(),[])
        
        Mparts = []
        coords = S.coords
        edg = S.getEdges()
        fac = S.getElemEdges()
        ele = S.elems
        d = S.distanceFromPlane(p,n)

        # Get the edges intersecting with the plane: 1 up and 1 down vertex
        d_edg = d[edg]
        edg_1_up = (d_edg > 0.).sum(axis=1) == 1
        edg_1_do = (d_edg < 0.).sum(axis=1) == 1
        w = edg_1_up * edg_1_do
        ind = where(w)[0]
        # compute the intersection points
        if ind.size != 0:
            rev = inverseUniqueIndex(ind)
            M = Mesh(S.coords,edg[w])
            x = M.toFormex().intersectionWithPlane(p,n).reshape(-1,3)
        
        # For each triangle, compute the number of cutting edges
        cut = w[fac]
        ncut = cut.sum(axis=1)

        # Split the triangles based on the number of inside vertices
        d_ele = d[ele]
        ins = d_ele == 0.
        nins = ins.sum(axis=1)
        ins0,ins1,ins2,ins3 = [ where(nins==i)[0] for i in range(4) ]

        # No inside vertices -> 2 cutting edges
        if ins0.size > 0:
            cutedg = fac[ins0][cut[ins0]].reshape(-1,2)
            e0 = rev[cutedg]
            Mparts.append(Mesh(x,e0).compact())

        # One inside vertex
        if ins1.size > 0:
            ncut1 = ncut[ins1]
            cut10,cut11 = [ where(ncut1==i)[0] for i in range(2) ]
            # 0 cutting edges: does not generate a line segment
            # 1 cutting edge
            if cut11.size != 0:
                e11ins = ele[ins1][cut11][ins[ins1][cut11]].reshape(-1,1)
                cutedg = fac[ins1][cut11][cut[ins1][cut11]].reshape(-1,1)
                e11cut = rev[cutedg]
                x11 = Coords.concatenate([coords,x],axis=0)
                e11 = concatenate([e11ins,e11cut+len(coords)],axis=1)
                Mparts.append(Mesh(x11,e11).compact())

        # Two inside vertices -> 0 cutting edges
        if ins2.size > 0:
            e2 = ele[ins2][ins[ins2]].reshape(-1,2)
            Mparts.append(Mesh(coords,e2).compact())

        # Three inside vertices -> 0 cutting edges
        if ins3.size > 0:
            insedg =  fac[ins3].reshape(-1)
            insedg.sort(axis=0)
            double = insedg == roll(insedg,1,axis=0)
            insedg = setdiff1d(insedg,insedg[double])
            if insedg.size != 0:
                e3 = edg[insedg]
                Mparts.append(Mesh(coords,e3).compact())

        # Done with getting the segments
        if len(Mparts) ==  0:
            # No intersection: return empty mesh
            return Mesh(Coords(),[])
        elif len(Mparts) == 1:
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


    def slice(self,dir=0,nplanes=20):
        """Intersect a surface with a sequence of planes.

        A sequence of nplanes planes with normal dir is constructed
        at equal distances spread over the bbox of the surface.

        The return value is a list of intersectionWithPlane() return
        values, i.e. a list of Meshes, one for every cutting plane.
        In each Mesh the simply connected parts are identified by
        property number.
        """
        o = self.center()
        if type(dir) is int:
            dir = unitVector(dir)
        xmin,xmax = self.coords.directionalExtremes(dir,o)
        P = Coords.interpolate(xmin,xmax,nplanes)
        return [ self.intersectionWithPlane(p,dir) for p in P ]

##################  Smooth a surface #############################

    
    def smooth(self,method='lowpass',iterations=1,lambda_value=0.5,neighbourhood=1,alpha=0.0,beta=0.2,verbose=False):
        """Smooth the surface.

        Returns a TriSurface which is a smoothed version of the original.
        Three smoothing methods are available: 'lowpass', 'laplace', and
        'gts'. The first two are built-in, the latter uses the external
        command `gtssmooth`.

        Parameters:
        
        - `method`: 'lowpass', 'laplace', or 'gts'
        - `iterations`: int: number of iterations
        - `lambda_value`: float: lambda value used in the filters

        Extra parameters for 'lowpass' and 'laplace':
        
        - `neighbourhood`: int: maximum number of edges to follow to define
          node neighbourhood

        Extra parameters for 'laplace':
        
        - `alpha`, `beta`: float: parameters for the laplace method.

        Extra parameters for 'gts':
        
        - `verbose`: boolean: requests more verbose output of the `gtssmooth`
          command

        Returns: the smoothed TriSurface
        """
        method = method.lower()
        if method == 'gts':
            return self._gts_smooth(iterations,lambda_value,verbose)

        # find adjacency
        adj = adjacencyArrays(self.getEdges(),nsteps=neighbourhood)[1:]
        adj = column_stack(adj)
        # find interior vertices
        bound_edges = self.borderEdgeNrs()
        inter_vertex = resize(True,self.ncoords())
        inter_vertex[unique(self.getEdges()[bound_edges])] = False
        # calculate weights
        w = ones(adj.shape,dtype=float)
        w[adj<0] = 0.
        val = (adj>=0).sum(-1).reshape(-1,1)
        w /= val
        w = w.reshape(adj.shape[0],adj.shape[1],1)

        # recalculate vertices
        
        if method == 'laplace':
            xo = self.coords
            x = self.coords.copy()
            for step in range(iterations):
                xn = x + lambda_value*(w*(x[adj]-x.reshape(-1,1,3))).sum(1)
                xd = xn - (alpha*xo + (1-alpha)*x)
                x[inter_vertex] = xn[inter_vertex] - (beta*xd[inter_vertex] + (1-beta)*(w[inter_vertex]*xd[adj[inter_vertex]]).sum(1))

        else: # default: lowpass
            k = 0.1
            mu_value = -lambda_value/(1-k*lambda_value)
            x = self.coords.copy()
            for step in range(iterations):
                x[inter_vertex] = x[inter_vertex] + lambda_value*(w[inter_vertex]*(x[adj[inter_vertex]]-x[inter_vertex].reshape(-1,1,3))).sum(1)
                x[inter_vertex] = x[inter_vertex] + mu_value*(w[inter_vertex]*(x[adj[inter_vertex]]-x[inter_vertex].reshape(-1,1,3))).sum(1)
                
        return TriSurface(x,self.elems,prop=self.prop)


    def smoothLowPass(self,iterations=2,lambda_value=0.5,neighbours=1):
        return self.smooth('lowpass',iterations/2,lambda_value,neighbours)


    def smoothLaplaceHC(self,iterations=2,lambda_value=0.5,alpha=0.,beta=0.2,neighbours=1):
        return self.smooth('lowpass',iterations,lambda_value,neighbours,alpha,beta)



###################### Methods using admesh/GTS #############################
 
 
    def fixNormals(self):
        """Fix the orientation of the normals.

        Some surface operations may result in improperly oriented normals.
        This tries to reverse improperly oriented normals so that a
        single oriented surface is achieved. It only works on a
        closed surface.

        In the current version, this uses the external program `admesh`,
        so this should be installed on the machine.

        If the surface was a (possibly non-orientable) manifold, the result
        will be an orientable manifold. This is a necessary condition
        for the `gts` methods to be applicable.
        """
        if self.nelems() == 0:
            # Protect against impossible file handling for empty surfaces
            return self
        tmp = tempfile.mktemp('.stl')
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'stl')
        tmp1 = tempfile.mktemp('.off')
        cmd = "admesh -d --write-off='%s' '%s'" % (tmp1,tmp)
        pf.message("Fixing surface normals with command\n %s" % cmd)
        sta,out = runCommand(cmd)
        pf.message("Reading result from %s" % tmp1)
        S = TriSurface.read(tmp1)   
        os.remove(tmp)
        os.remove(tmp1)    
        return S.setProp(self.prop)


    def check(self,verbose=False):
        """Check the surface using gtscheck.

        Checks whether the surface is a closed, orientable,
        non self-intersecting manifold. This is a necessary condition
        for the use of the `gts` methods: split, coarsen, refine, boolean.

        Returns 0 if the surface passes the tests, nonzero if not.
        A full report is printed out. 

        The :meth:`fixNormals` and :meth:`reverse` methods may be used to fix
        the normals of an otherwise correct closed manifold.
        """
        cmd = 'gtscheck'
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Checking with command\n %s" % cmd)
        cmd += ' < %s' % tmp
        sta,out = runCommand(cmd,False)
        os.remove(tmp)
        pf.message(out)
        if sta == 0:
            pf.message('The surface is a closed, orientable non self-intersecting manifold')
        return sta
 

    def split(self,base,verbose=False):
        """Split the surface using gtssplit.

        Splits the surface into connected and manifold components.
        This uses the external program `gtssplit`. The surface
        should be a closed orientable non-intersecting manifold.
        Use the :meth:`check` method to find out.

        This method creates a series of files with given base name,
        each file contains a single connected manifold.
        """
        cmd = 'gtssplit -v %s' % base
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Splitting with command\n %s" % cmd)
        cmd += ' < %s' % tmp
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            pf.message(out)
        #
        # WE SHOULD READ THIS FILES BACK !!!
        #
   

    def coarsen(self,min_edges=None,max_cost=None,
                mid_vertex=False, length_cost=False, max_fold=1.0,
                volume_weight=0.5, boundary_weight=0.5, shape_weight=0.0,
                progressive=False, log=False, verbose=False):
        """Coarsen the surface using gtscoarsen.

        Construct a coarsened version of the surface.
        This uses the external program `gtscoarsen`. The surface
        should be a closed orientable non-intersecting manifold.
        Use the :meth:`check` method to find out.

        Parameters:
        
        - `min_edges`: int: stops the coarsening process if the number of
          edges was to fall below it
        - `max_cost`: float: stops the coarsening process if the cost of
          collapsing an edge is larger
        - `mid_vertex`: boolean: use midvertex as replacement vertex instead
          of the default, which is a volume optimized point
        - `length_cost`: boolean: use length^2 as cost function instead of the
          default optimized point cost
        - `max_fold`: float: maximum fold angle in degrees
        - `volume_weight`: float: weight used for volume optimization
        - `boundary_weight`: float: weight used for boundary optimization
        - `shape_weight`: float: weight used for shape optimization
        - `progressive`: boolean: write progressive surface file
        - `log`: boolean: log the evolution of the cost
        - `verbose`: boolean: print statistics about the surface
        """
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
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Coarsening with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            pf.message(out)
        pf.message("Reading coarsened model from %s" % tmp1)
        S = TriSurface.read(tmp1)        
        os.remove(tmp1)
        return S
   

    def refine(self,max_edges=None,min_cost=None,log=False,verbose=False):
        """Refine the surface using gtsrefine.

        Construct a refined version of the surface.
        This uses the external program `gtsrefine`. The surface
        should be a closed orientable non-intersecting manifold.
        Use the :meth:`check` method to find out.

        Parameters:
        
        - `max_edges`: int: stop the refining process if the number of
          edges exceeds this value
        - `min_cost`: float: stop the refining process if the cost of refining
          an edge is smaller
        - `log`: boolean: log the evolution of the cost
        - `verbose`: boolean: print statistics about the surface
        """
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
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Refining with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            pf.message(out)
        pf.message("Reading refined model from %s" % tmp1)
        S = TriSurface.read(tmp1)        
        os.remove(tmp1)
        return S


    def _gts_smooth(self,iterations=1,lambda_value=0.5,verbose=False):
        """Smooth the surface using gtssmooth.

        Smooth a surface by applying iterations of a Laplacian filter.
        This uses the external program `gtssmooth`. The surface
        should be a closed orientable non-intersecting manifold.
        Use the :meth:`check` method to find out.

        Parameters:

        - `lambda_value`: float: Laplacian filter parameter
        - `iterations`: int: number of iterations
        - `verbose`: boolean: print statistics about the surface

        See also: :meth:`smoothLowPass`, :meth:`smoothLaplaceHC`
        """
        cmd = 'gtssmooth'
#        if fold_smoothing:
#            cmd += ' -f %s' % fold_smoothing
        cmd += ' %s %s' % (lambda_value,iterations)
        if verbose:
            cmd += ' -v'
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Smoothing with command\n %s" % cmd)
        cmd += ' < %s > %s' % (tmp,tmp1)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        if sta or verbose:
            pf.message(out)
        pf.message("Reading smoothed model from %s" % tmp1)
        S = TriSurface.read(tmp1)        
        os.remove(tmp1)
        return S


    def boolean(self,surf,op,intersection_curve=False,check=False,verbose=False):
        """Perform a boolean operation with another surface.

        Boolean operations between surfaces are a basic operation in
        free surface modeling. Both surfaces should be closed orientable
        non-intersecting manifolds.
        Use the :meth:`check` method to find out.

        The boolean operations are set operations on the enclosed volumes:
        union('+'), difference('-') or intersection('*').

        Parameters:

        - `intersection_curve`: boolean: output an OOGL (Geomview)
          representation of the curve intersection of the surfaces
        - `check`: boolean: check that the surfaces are not self-intersecting;
          if one of them is, the set of self-intersecting faces is written
          (as a GtsSurface) on standard output
        - `verbose`: boolean: print statistics about the surface
        """
        cmd = 'gtsset'
        ops = {'+':'union', '-':'diff', '*':'inter'}
        if intersection_curve:
            cmd += ' -i'
        if check:
            cmd += ' -s'
        if verbose:
            cmd += ' -v'
        cmd += ' '+ops[op]
        tmp = tempfile.mktemp('.gts')
        tmp1 = tempfile.mktemp('.gts')
        tmp2 = tempfile.mktemp('.stl')
        pf.message("Writing temp file %s" % tmp)
        self.write(tmp,'gts')
        pf.message("Writing temp file %s" % tmp1)
        surf.write(tmp1,'gts')
        pf.message("Performing boolean operation with command\n %s" % cmd)
        cmd += ' %s %s | gts2stl > %s' % (tmp,tmp1,tmp2)
        sta,out = runCommand(cmd)
        os.remove(tmp)
        os.remove(tmp1)
        if sta or verbose:
            pf.message(out)
        pf.message("Reading result from %s" % tmp2)
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
    pf.message("Transforming .OFF model %s to tetgen .smesh" % fn)
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
    pf.message("Removing %s out of %s triangles" % (remove.shape[0],elems.shape[0]))
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
    pf.message("Actually removed %s triangles, leaving %s" % (nelems-mag1.shape[0],elems.shape[0]))

    return elems

#####################################################################
### Some simple surfaces ###

def Rectangle(nx,ny):
    """Create a plane rectangular surface consisting of a nx,ny grid."""
    F = Formex(mpattern('12-34')).replic2(nx,ny,1,1)    
    return TriSurface(F)

def Cube():
    """Create the surface of a cube

    Returns a TriSurface representing the surface of a unit cube.
    Each face of the cube is represented by two triangles.
    """
    back = Formex(mpattern('12-34'))
    fb = back.reverse() + back.translate(2,1)
    faces = fb + fb.rollAxes(1) + fb.rollAxes(2)
    return TriSurface(faces)


def Sphere(level=4,verbose=False,filename=None):
    """Create a spherical surface by calling the gtssphere command.

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
    pf.message("Writing file %s" % tmp)
    sta,out = runCommand(cmd)
    if sta or verbose:
        pf.message(out)
    pf.message("Reading model from %s" % tmp)
    S = TriSurface.read(tmp)
    if filename is None:
        os.remove(tmp)
    return S


####### Unsupported functions needing cleanup #######################

from utils import deprecation

def checkDistanceLinesPointsTreshold(p, q, m, dtresh):
    """_p are np points, q m are nl lines, dtresh are np distances. It returns the indices of lines, points which are in a distance < dtresh. The distance point-line is calculated using Pitagora as it seems the fastest way."""
    cand=[]
    m=normalize(m)
    for i in range(q.shape[0]):
        hy= p-q[i]
        dpl=(abs(length(hy)**2.-dotpr(hy, m[i])**2.))**0.5
        wd= dpl<=dtresh
        cand.append([i,where(wd)[0]] )
    candwl= concatenate([repeat(cand[i][0], cand[i][1].shape[0]) for i in range(len(cand))])
    candwt= concatenate([cand[i][1] for i in range(len(cand))])
    return candwl, candwt

def intersectLineWithPlaneOne2One(q,m,p,n):
    """_it returns for each pair of line(q,m) and plane (p,n) the point of intersection. plane: (x-p)n=0, line: x=q+t m. It find the scalar t and returns the point."""
    t=dotpr(n, (p-q))/dotpr(n, m)
    return q+m*t[:, newaxis]

def checkPointInsideTriangleOne2One(tpi, pi, atol=1.e-5):
    """_return a 1D boolean with the same dimension of tpi and pi. The value [i] is True if the point pi[i] is inside the triangle tpi[i]. It uses areas to check it. """
    tpi3= column_stack([tpi[:, 0], tpi[:, 1], pi, tpi[:, 0], pi, tpi[:, 2], pi, tpi[:, 1], tpi[:, 2]]).reshape(pi.shape[0]*3,  3, 3)
    #areas
    Atpi3=(length(cross(tpi3[:,1]-tpi3[:,0],tpi3[:,2]-tpi3[:,1]))*0.5).reshape(pi.shape[0], 3).sum(axis=1)#area and sum
    Atpi=length(cross(tpi[:,1]-tpi[:,0],tpi[:,2]-tpi[:,1]))*0.5#area
    return -(Atpi3>Atpi+atol)#True mean point inside triangle

def intersectSurfaceWithLines(ts, qli, mli, atol=1.e-5):
    """_it takes a TriSurface ts and a set of lines ql,ml and intersect the lines with the TriSurface.
    It returns the points of intersection and the indices of the intersected line and triangle.
    TODO: the slowest part is computing the distances of lines from triangles, can it be faster? """
    #find Bounding Sphere for each triangle
    tsr,  tsc, tsn=triangleBoundingCircle(ts.coords[ts.elems])
    wl, wt=checkDistanceLinesPointsTreshold(tsc,qli, mli, tsr)#slow part
    #find the intersection points xc only for the candidates wl,wt
    npl= ts.areaNormals()[1]
    xc= intersectLineWithPlaneOne2One(qli[wl],mli[wl],tsc[wt],npl[wt])
    #check if each intersection is really inside the triangle
    tsw=ts.select(wt)
    tsw=tsw.coords[tsw.elems]
    xIn=checkPointInsideTriangleOne2One(tsw, xc, atol)
    #takes only intersections that fall inside the triangle
    return xc[xIn], wl[xIn], wt[xIn]

def intersectSurfaceWithSegments(s1, segm, atol=1.e-5):
    """_it takes a TriSurface ts and a set of segments (-1,2,3) and intersect the segments with the TriSurface.
    It returns the points of intersections and, for each point, the indices of the intersected segment and triangle"""
    p, il, it=intersectSurfaceWithLines(s1, segm[:, 0], normalize(segm[:, 1]-segm[:, 0]),atol)
    win= length(p-segm[:, 0][il])+ length(p-segm[:, 1][il])< length(segm[:, 1][il]-segm[:, 0][il])+atol
    return p[win], il[win], it[win]


# For compatibility with older project files, this can be uncommented
# Surface = TriSurface

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print(a)
    
# End
