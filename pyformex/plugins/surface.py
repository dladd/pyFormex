# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import os
import globaldata as GD
from plugins import tetgen,connectivity
from utils import runCommand, changeExt,countLines,mtime,hasExternal
from formex import *
import tempfile

hasExternal('admesh')
hasExternal('tetgen')
#hasExternal('gts')


# Conversion of surface data models


def expandElems(elems):
    """Transform elems to edges and faces.

    elems is an (nelems,nplex) integer array of element node numbers.
    The maximum node number should be less than 2**31 or approx. 2 * 10**9 !!

    Return a tuple edges,faces where
    - edges is an (nedges,2) int32 array of edges connecting two node numbers.
    - faces is an (nelems,nplex) int32 array with the edge numbers connecting
      each pair os subsequent nodes in the elements of elems.

    The order of the edges respects the node order, and starts with nodes 0-1.
    The node numbering in the edges is always lowest node number first.

    The inverse operation is compactElems.
    """
    nelems,nplex = elems.shape
    magic = elems.max() + 1
    if magic > 2**31:
        raise RuntimeError,"Cannot compact edges for more than 2**31 nodes"
    n = arange(nplex)
    edg = column_stack([n,roll(n,-1)])
    alledges = elems[:,edg]
    # sort edge nodes with lowest number first
    alledges.sort()
    if GD.options.fastencode:
        edg = alledges.reshape((-1,2))
        codes = edg.view(int64)
    else:
        edg = alledges.astype(int64).reshape((-1,2))
        codes = edg[:,0] * magic + edg[:,1]
    # keep the unique edge numbers
    uniqid,uniq = unique1d(codes,True)
    # we suppose uniq is sorted 
    uedges = uniq.searchsorted(codes)
    edges = column_stack([uniq/magic,uniq%magic])
    faces = uedges.reshape((nelems,nplex))
    return edges,faces


def compactElems(edges,faces):
    """Return compacted elems form edges and faces.

    This is the inverse operation of expandElems.
    """
    elems = edges[faces]
    flag1 = (elems[:,0,0]==elems[:,1,0]) + (elems[:,0,0]==elems[:,1,1])
    flag2 = (elems[:,2,0]==elems[:,1,0]) + (elems[:,2,0]==elems[:,1,1])
    nod0 = where(flag1,elems[:,0,1],elems[:,0,0])
    nod1 = where(flag1,elems[:,0,0],elems[:,0,1])
    nod2 = where(flag2,elems[:,2,0],elems[:,2,1])
    elems = column_stack([nod0,nod1,nod2])
    return elems


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
    print "Reading GTS file %s" % fn
    fil = file(fn,'r')
    header = fil.readline().split()
    ncoords,nedges,nfaces = map(int,header[:3])
    if len(header) >= 7 and header[6].endswith('Binary'):
        sep=''
    else:
        sep=' '
    coords = fromfile(file=fil, dtype=Float, count=3*ncoords, sep=sep)
    edges = fromfile(file=fil, dtype=int32, count=2*nedges, sep=' ')
    faces = fromfile(file=fil, dtype=int32, count=3*nfaces, sep=' ')
    print "Read %d coords, %d edges, %d faces" % (ncoords,nedges,nfaces)
    return coords.reshape((-1,3)),\
           edges.reshape((-1,2)) - 1,\
           faces.reshape((-1,3)) - 1


def read_off(fn):
    """Read an OFF surface mesh.

    The mesh should consist of only triangles!
    Returns a nodes,elems tuple.
    """
    print "Reading .OFF %s" % fn
    fil = file(fn,'r')
    head = fil.readline().strip()
    if head != "OFF":
        print "%s is not an OFF file!" % fn
        return None,None
    nnodes,nelems,nedges = map(int,fil.readline().split())
    nodes = fromfile(file=fil, dtype=Float, count=3*nnodes, sep=' ')
    # elems have number of vertices + 3 vertex numbers
    elems = fromfile(file=fil, dtype=int32, count=4*nelems, sep=' ')
    print "Read %d nodes and %d elems" % (nnodes,nelems)
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
    runCommand("%s/external/gambit-neu '%s'" % (GD.cfg['pyformexdir'],fn))
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
    print "The model contains %d degenerate triangles" % degen.shape[0]
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
    print "Cannot write binary STL files yet!" % fn
    pass


def write_gambit_neutral(fn,nodes,elems):
    print "Cannot write Gambit neutral files yet!" % fn
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

    For each element of Formex, return the volume of the tetrahedron
    formed by the point pt (default the center of x) and the 3 points
    of the element.
    """
    x -= x.center()
    a,b,c = [ x[:,i,:] for i in range(3) ]
    d = cross(b,c)
    e = (a*d).sum(axis=-1)
    v = sign(e) * abs(e)/6.
    return v


############################################################################
# The Surface class

def coordsmethod(f):
    """Define a Surface method as the equivalent Coords method.

    This decorator replaces the Surface's vertex coordinates with the
    ones resulting from applying the transform f.
    
    The coordinates are changed inplane, so copy them before if you do not
    want them to be lost.
    """
    def newf(self,*args,**kargs):
        repl = getattr(Coords,f.__name__)
        self.coords = repl(self.coords,*args,**kargs)
        newf.__name__ = f.__name__
        newf.__doc__ = repl.__doc__
    return newf


class Surface(object):
    """A class for handling triangulated 3D surfaces."""

    def __init__(self,*args):
        """Create a new surface.

        The surface contains ntri triangles, each having 3 vertices with
        3 coordinates.
        The surface can be initialized from one of the following:
        - a (ntri,3,3) shaped array of floats ;
        - a 3-plex Formex with ntri elements ;
        - an (ncoords,3) float array of vertex coordinates and
          an (ntri,3) integer array of vertex numbers ;
        - an (ncoords,3) float array of vertex coordinates,
          an (nedges,2) integer array of vertex numbers,
          an (ntri,3) integer array of edges numbers.

        Internally, the surface is stored in a (coords,edges,faces) tuple.
        """
        self.coords = self.edges = self.faces = None
        self.elems = None
        self.areas = None
        self.normals = None
        self.conn = None
        self.adj = None
        if hasattr(self,'edglen'):
            del self.edglen
        self.p = None
        if len(args) == 0:
            return
        if len(args) == 1:
            # a Formex/STL model
            a = args[0]
            if not isinstance(a,Formex):
                a = Formex(a)
            if a.nplex() != 3:
                raise ValueError,"Expected a plex-3 Formex"
            self.coords,self.elems = a.feModel()
            self.p = a.p
            self.refresh()

        else:
            a = Coords(args[0])
            if len(a.shape) != 2:
                raise ValueError,"Expected a 2-dim coordinates array"
            self.coords = a
            
            a = asarray(args[1])
            if a.dtype.kind != 'i' or a.ndim != 2 or a.shape[1]+len(args) != 5:
                raise "Got invalid second argument"
            if a.max() >= self.coords.shape[0]:
                raise ValueError,"Some vertex number is too high"
            if len(args) == 2:
                self.elems = a
                self.refresh()
            elif len(args) == 3:
                self.edges = a

                a = asarray(args[2])
                if not (a.dtype.kind == 'i' and a.ndim == 2 and a.shape[1] == 3):
                    raise "Got invalid third argument"
                if a.max() >= self.edges.shape[0]:
                    raise ValueError,"Some edge number is too high"
                self.faces = a

            else:
                raise RuntimeError,"Too many arguments"
 

    # To keep the data consistent:
    # ANY function that uses self.elems should call self.refresh()
    #     BEFORE using it.
    # ANY function that changes self.elems should
    #     - invalidate self.edges and/or self.faces by setting it to None,
    #     - call self.refresh() AFTER changing it.

    def refresh(self):
        """Make the internal information consistent and complete.

        This function should be called after one of the data fields
        have been changed.
        """
        if self.coords is None:
            return
        if type(self.coords) != Coords:
            self.coords = Coords(self.coords)
        if self.edges is None or self.faces is None:
            self.edges,self.faces = expandElems(self.elems)
        if self.elems is None:
            self.elems = compactElems(self.edges,self.faces)

###########################################################################
    #
    #   Return information about a Surface
    #

    def ncoords(self):
        return self.coords.shape[0]

    def nedges(self):
        return self.edges.shape[0]

    def nfaces(self):
        return self.faces.shape[0]

    # The following are defined to get a unified interface with Formex
    nelems = nfaces
   
    def nplex(self):
        return 3
    
    def ndim(self):
        return 3

    npoints = ncoords
    
    def shape(self):
        """Return the number of ;points, edges, faces of the Surface."""
        return self.coords.shape[0],self.edges.shape[0],self.faces.shape[0]


    # Properties
    def setProp(self,p=None):
        """Create or delete the property array for the Surface.

        A property array is a rank-1 integer array with dimension equal
        to the number of elements in the Surface.
        You can specify a single value or a list/array of integer values.
        If the number of passed values is less than the number of elements,
        they wil be repeated. If you give more, they will be ignored.
        
        If a value None is given, the properties are removed from the Surface.
        """
        if p is None:
            self.p = None
        else:
            p = array(p).astype(Int)
            self.p = resize(p,(self.nelems(),))
        return self

    def prop(self):
        """Return the properties as a numpy array (ndarray)"""
        return self.p

    def maxprop(self):
        """Return the highest property value used, or None"""
        if self.p is None:
            return None
        else:
            return self.p.max()

    def propSet(self):
        """Return a list with unique property values."""
        if self.p is None:
            return None
        else:
            return unique(self.p)

    # The following functions get the corresponding information from
    # the underlying Coords object

    def x(self):
        return self.coords.x()
    def y(self):
        return self.coords.y()
    def z(self):
        return self.coords.z()

    def bbox(self):
        return self.coords.bbox()

    def center(self):
        return self.coords.center()

    def centroid(self):
        return self.coords.centroid()

    def sizes(self):
        return self.coords.sizes()

    def diagonal(self):
        return self.coords.diagonal()

    def bsphere(self):
        return self.coords.bsphere()


    def centroids(self):
        """Return the centroids of all elements of the Formex.

        The centroid of an element is the point whose coordinates
        are the mean values of all points of the element.
        The return value is an (nfaces,3) shaped Coords array.
        """
        return self.toFormex().centroids()

    #  Distance

    def distanceFromPlane(self,*args,**kargs):
        return self.coords.distanceFromPlane(*args,**kargs)

    def distanceFromLine(self,*args,**kargs):
        return self.coords.distanceFromLine(*args,**kargs)


    def distanceFromPoint(self,*args,**kargs):
        return self.coords.distanceFromPoint(*args,**kargs)
 

    # Data conversion
    
    def feModel(self):
        """Return a tuple of nodal coordinates and element connectivity."""
        self.refresh()
        return self.coords,self.elems


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
            return Surface(*read_off(fn))
        elif ftype == 'gts':
            #print "READING GTS"
            ret = read_gts(fn)
            #print ret
            S = Surface(*ret)
            #print S.shape()
            return S
        elif ftype == 'stl':
            return Surface(*read_stl(fn))
        elif ftype == 'neu':
            return Surface(*read_gambit_neutral(fn))
        elif ftype == 'smesh':
            return Surface(*tetgen.readSurface(fn))
        else:
            raise "Unknown Surface type, cannot read file %s" % fn


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
            write_gts(fname,self.coords,self.edges,self.faces)
            GD.message("Wrote %s vertices, %s edges, %s faces" % self.shape())
        elif ftype in ['stl','off','neu','smesh']:
            self.refresh()
            if ftype == 'stl':
                write_stla(fname,self.coords[self.elems])
            elif ftype == 'off':
                write_off(fname,self.coords,self.elems)
            elif ftype == 'neu':
                write_gambit_neutral(fname,self.coords,self.elems)
            elif ftype == 'smesh':
                write_smesh(fname,self.coords,self.elems)
            GD.message("Wrote %s vertices, %s elems" % (self.ncoords(),self.nfaces()))
        else:
            print "Cannot save Surface as file %s" % fname


    def toFormex(self):
        """Convert the surface to a Formex."""
        self.refresh()
        return Formex(self.coords[self.elems],self.p)


    @coordsmethod
    def scale(self,*args,**kargs):
        pass
    @coordsmethod
    def translate(self,*args,**kargs):
        pass
    @coordsmethod
    def rotate(self,*args,**kargs):
        pass
    @coordsmethod
    def shear(self,*args,**kargs):
        pass
    @coordsmethod
    def reflect(self,*args,**kargs):
        pass
    @coordsmethod
    def affine(self,*args,**kargs):
        pass
 

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
            cmd += ' -c %d' % max_cost
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
            cmd += ' -c %d' % min_cost
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


    def areaNormals(self):
        """Compute the area and normal vectors of the surface triangles.

        The normal vectors are normalized.
        The area is always positive.

        The values are returned and saved in the object.
        """
        if self.areas is None or self.normals is None:
            self.refresh()
            self.areas,self.normals = areaNormals(self.coords[self.elems])
        return self.areas,self.normals


    def facetArea(self):
        return self.areaNormals()[0]
        

    def area(self):
        """Return the area of the surface"""
        area = self.areaNormals()[0]
        return area.sum()


    def volume(self):
        self.refresh()
        x = self.coords[self.elems]
        return surface_volume(x).sum()


    def connections(self):
        """Find the elems connected to edges."""
        if self.conn is None:
            self.conn = connectivity.reverseIndex(self.faces)
        return self.conn
    

    def nConnected(self):
        """Find the number of elems connected to edges."""
        return (self.connections() >=0).sum(axis=-1)


    def adjacency(self):
        """Find the elems adjacent to elems."""
        if self.adj is None:
            nfaces = self.nfaces()
            rfaces = connectivity.reverseIndex(self.faces)
            # this gives all adjacent elements including element itself
            adj = rfaces[self.faces].reshape((nfaces,-1))
            fnr = arange(nfaces).reshape((nfaces,-1))
            # remove the element itself
            self.adj = adj[adj != fnr].reshape((nfaces,-1))
        return self.adj


    def nAdjacent(self):
        """Find the number of adjacent elems."""
        return (self.adjacency() >=0).sum(axis=-1)


    def surfaceType(self):
        ncon = self.nConnected()
        nadj = self.nAdjacent()
        maxcon = ncon.max()
        mincon = ncon.min()
        manifold = maxcon == 2
        closed = mincon == 2
        return manifold,closed,mincon,maxcon


    def borderEdges(self):
        """Detect the border elements of Surface.

        The border elements are the edges having less than 2 connected elements.
        Returns a list of edge numbers.
        """
        return self.nConnected() <= 1
        

    def isManifold(self):
        return self.surfaceType()[0] 


    def isClosedManifold(self):
        stype = self.surfaceType()
        return stype[0] and stype[1]
    

    def border(self):
        """Return the border of Surface as a Plex-2 Formex.

        The border elements are the edges having less than 2 connected elements.
        Returns a list of edge numbers.
        """
        return Formex(self.coords[self.edges[self.borderEdges()]])


    def edgeCosAngles(self):
        """Return the cos of the angles over all edges.
        
        The surface should be a manifold (max. 2 elements per edge).
        Edges with only one element get angles = 1.0.
        """
        conn = self.connections()
        # Bail out if some edge has more than two connected faces
        if conn.shape[1] != 2:
            raise RuntimeError,"Surface is not a manifold"
        angles = ones(self.nedges())
        conn2 = (conn >= 0).sum(axis=-1) == 2
        n = self.areaNormals()[1][conn[conn2]]
        angles[conn2] = dotpr(n[:,0],n[:,1])
        return angles


    def edgeAngles(self):
        """Return the angles over all edges (in degrees)."""
        return arccos(self.edgeCosAngles()) / rad


    def data(self):
        """Compute data for all edges and faces."""
        if hasattr(self,'edglen'):
            return
        self.areaNormals()
        edg = self.coords[self.edges]
        edglen = length(edg[:,1]-edg[:,0])
        facedg = edglen[self.faces]
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
        print  (
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
    

    def partitionByFront(self,okedges,firstprop=0,startat=0,maxruns=-1,check=True):
        """Detects different parts of the surface using a frontal method.

        func is a function that takes the edge connection table as input and
        produces an array with nedges values flagging with a True/nonzero
        value all edges where the connected elements should belong to the
        same part.
        """
        p = -ones((self.nfaces()),dtype=int)
        if self.nfaces() <= 0:
            return p
        # Construct table of elements connected to each edge
        conn = self.connections()
        # Bail out if some edge has more than two connected faces
        if conn.shape[1] != 2:
            GD.warning("Surface is not a manifold")
            return p
        # Check size of okedges
        if okedges.ndim != 1 or okedges.shape[0] != self.nedges():
            raise ValueError,"okedges has incorrect shape"

        # Remember edges left for processing
        todo = ones((self.nedges(),),dtype=bool)
        startat = clip(startat,0,self.nfaces())
        elems = array([startat])
        prop = max(0,firstprop)
        run = 0
        while elems.size > 0 and (maxruns < 0 or run < maxruns):
            run += 1
            # Store prop value
            p[elems] = prop
            # Determine border
            edges = unique(self.faces[elems])
            edges = edges[todo[edges]]

            if edges.size > 0:
                # flag edges as done
                todo[edges] = 0
                # take elements connected over small angle
                elems = conn[edges][okedges[edges]].ravel()
                if elems.size > 0:
                    continue

            # No more elements in this part: start a new one
            elems = where(p<firstprop)[0]
            if elems.size > 0:
                # Start a new part
                elems = elems[[0]]
                prop += 1
                
        return p


    def partitionByConnection(self):
        okedges = ones((self.nedges()),dtype=bool)
        return self.partitionByFront(okedges)


    def partitionByAngle2(self,angle=180.,firstprop=0,startat=0,maxruns=-1):
        conn = self.connections()
        # Flag edges that connect two faces
        conn2 = (conn >= 0).sum(axis=-1) == 2
        # compute normals and flag small angles over edges
        cosangle = cosd(angle)
        n = self.areaNormals()[1][conn[conn2]]
        small_angle = ones(conn2.shape,dtype=bool)
        small_angle[conn2] = dotpr(n[:,0],n[:,1]) >= cosangle
        return self.partitionByFront(small_angle)
        

    def partitionByAngle(self,angle,firstprop=0,startat=0,maxruns=-1):
        """Detects different parts of the surface.
        
        Faces are considered to belong to the same part if the angle between
        these faces is smaller than the given value.
        """
        p = -ones((self.nfaces()),dtype=int)
        if self.nfaces() <= 0:
            return p
        # Construct table of elements connected to each edge
        conn = self.connections()
        # Bail out if some edge has more than two connected faces
        if conn.shape[1] != 2:
            GD.warning("Surface is not a manifold")
            return p

        # Flag edges that connect two faces
        conn2 = (conn >= 0).sum(axis=-1) == 2
        # compute normals and flag small angles over edges
        cosangle = cosd(angle)
        n = self.areaNormals()[1][conn[conn2]]
        small_angle = ones(conn2.shape,dtype=bool)
        small_angle[conn2] = dotpr(n[:,0],n[:,1]) >= cosangle

        # Remember edges and elements left for processing
        todo = ones((self.nedges(),),dtype=bool)

        # start with element startat
        startat = clip(startat,0,self.nfaces())
        elems = array([startat])
        prop = max(0,firstprop)
        run = 0
        while elems.size > 0 and (maxruns < 0 or run < maxruns):
            run += 1
            # Store prop value
            p[elems] = prop
            # Determine border
            edges = unique(self.faces[elems])
            edges = edges[todo[edges]]

            if edges.size > 0:
                # flag edges as done
                todo[edges] = 0
                # take elements connected over small angle
                elems = conn[edges][small_angle[edges]].ravel()
                if elems.size > 0:
                    continue

            # No more elements in this part: start a new one
            elems = where(p<firstprop)[0]
            if elems.size > 0:
                # Start a new part
                elems = elems[[0]]
                prop += 1
                
        return p



    def cutAtPlane(self,*args):
        """Cut a surface with a plane."""
        self.__init__(self.toFormex().cutAtPlane(*args))
        

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


def magic_numbers(elems,magic):
    elems = elems.astype(int64)
    elems.sort(axis=1)
    mag = ( elems[:,0] * magic + elems[:,1] ) * magic + elems[:,2]
    return mag


def demagic(mag,magic):
    first2,third = mag / magic, mag % magic
    first,second = first2 / magic, first2 % magic
    return column_stack([first,second,third]).astype(int32)


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

    mag1 = magic_numbers(elems,magic)
    mag2 = magic_numbers(triangles,magic)

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
    #print elems,remove
    GD.message("Removing %s out of %s triangles" % (remove.shape[0],elems.shape[0]))
    magic = elems.max()+1

    mag1 = magic_numbers(elems,magic)
    mag2 = magic_numbers(remove,magic)

    mag1.sort()
    mag2.sort()

    nelems = mag1.shape[0]

    pos = mag1.searchsorted(mag2)
    mag1[pos] = -1
    mag1 = mag1[mag1 >= 0]

    elems = demagic(mag1,magic)
    GD.message("Actually removed %s triangles, leaving %s" % (nelems-mag1.shape[0],elems.shape[0]))

    return elems


if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print a
    
# End
