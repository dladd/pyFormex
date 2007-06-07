#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import os
import globaldata as GD
from plugins import tetgen
from utils import runCommand, changeExt,countLines,mtime,hasExternal
from formex import *

hasExternal('admesh')
hasExternal('tetgen')

# The Stl class should not be used yet! Use the functions instead.
class STL(object):
    """A 3D surface described by a set of triangles."""

    def __init__(self,*args):
        """Create a new STL surface.

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
        """
        self.coords = self.nodes = self.elems = self.edges = self.faces = None
        if len(args) == 2:
            nodes = asarray(arg[0])
            elems = asarray(arg[1])
            if nodes.dtype.kind == 'f' and elems.dtype.kind == 'i' and \
                   nodes.ndim == 2 and elems.ndim == 2 and \
                   nodes.shape[1] == 3 and elems.shape[1] == 3:
                self.nodes = nodes
                self.elems = elems
            else:
                raise RuntimeError,"Invalid STL initialization data"


def areaNormals(x):
    """Compute the area and normal vectors of the triangles in x[n,3,3].

    The normal vectors are normalized.
    The area is always positive.
    """
    area,normals = vectorPairAreaNormals(x[:,1]-x[:,0],x[:,2]-x[:,1])
    return 0.5 * area, normals


def degenerate(area,norm):
    """Return a list of the degenerate faces according to area and normals.

    A face is degenerate if its surface is less or equal to zero or the
    normal has a nan.
    """
    return unique(concatenate([where(area<=0)[0],where(isnan(norm))[0]]))


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

    own = type(f) == str
    if own:
        f = file(f,'w')
    f.write("solid  Created by %s\n" % GD.Version)
    a,n = areaNormals(x)
    degen = degenerate(a,n)
    print "The model contains %d degenerate triangles" % degen.shape[0]
    for e,n in zip(x,v):
        f.write("  facet normal %s %s %s\n" % tuple(n))
        f.write("    outer loop\n")
        for p in e:
            f.write("      vertex %s %s %s\n" % tuple(p))
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write("endsolid\n")
    if own:
        f.close()


def read_error(cnt,line):
    """Raise an error on reading the stl file."""
    raise RuntimeError,"Invalid .stl format while reading line %s\n%s" % (cnt,line)


def read_stla(fn,dtype=Float,large=False,guess=True,off=False):
    """Read an ascii .stl file into an [n,3,3] float array.

    If the .stl is large, read_ascii_large() is recommended, as it is
    a lot faster.
    """
    if off:
        offname = stl_to_off(fn,sanitize=False)
        if offname:
            nodes,elems = read_off(offname)
            if not nodes is None:
                return nodes[elems]
        large=True
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


def read_stl(fn,sanitize=False):
    """Read an .stl file into an (nodes,elems) femodel.

    This is done by first coverting the .stl to .off format.
    """
    gtsname = stl_to_gts(fn)
    nodes,edges,faces = read_gts(gtsname)
    elems = expandEdges(edges,faces)
    return nodes,elems


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


def write_gambit_neutral(fn,nodes,elems):
    print "Cannot write file %s" % fn
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


def read_gts(fn):
    """Read a GTS surface mesh.

    Returns a nodes,edges,faces tuple.
    """
    print "Reading GTS file %s" % fn
    fil = file(fn,'r')
    header = fil.readline().split()
    nnodes,nedges,nfaces = map(int,header[:3])
    if len(header) >= 7 and header[6].endswith('Binary'):
        sep=''
    else:
        sep=' '
    nodes = fromfile(file=fil, dtype=Float, count=3*nnodes, sep=sep)
    edges = fromfile(file=fil, dtype=int32, count=2*nedges, sep=' ')
    faces = fromfile(file=fil, dtype=int32, count=3*nfaces, sep=' ')
    print "Read %d nodes, %d edges, %d faces" % (nnodes,nedges,nfaces)
    return nodes.reshape((-1,3)),\
           edges.reshape((-1,2)) - 1,\
           faces.reshape((-1,3)) - 1


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


def write_smesh(fn,nodes,elems):
    tetgen.writeSurface(fn,nodes,elems)


def readSurface(fn,ftype=None):
    if ftype is None:
        ftype = os.path.splitext(fn)[1]  # deduce from extension
    ftype = ftype.strip('.').lower()
    if ftype == 'stl':
        ofn = changeExt(fn,'.gts')
        if (not os.path.exists(ofn)) or (mtime(ofn) < mtime(fn)):
            stl_to_gts(fn)
        nodes,edges,faces = read_gts(ofn)
        elems = expandEdges(edges,faces)
    elif ftype == 'off':
        nodes,elems = read_off(fn)
    elif ftype == 'neu':
        nodes,elems = read_gambit_neutral(fn)
    elif ftype == 'smesh':
        nodes,elems = tetgen.readSurface(fn)
    elif ftype == 'gts':
        nodes,edges,faces = read_gts(fn)
        elems = expandEdges(edges,faces)
    else:
        print "Cannot read file %s" % fn
    return nodes,elems


def writeSurface(fn,nodes,elems,ftype=None):
    if ftype is None:
        ftype = os.path.splitext(fn)[1]  # deduce from extension
    ftype = ftype.strip('.').lower()
    print "Writing %s vertices" % nodes.shape[0]
    print "Writing %s triangles to file %s" % (elems.shape[0],fn)
    if ftype == 'stl':
        write_stla(fn,nodes[elems])
    elif ftype == 'off':
        write_off(fn,nodes,elems)
    elif ftype == 'neu':
        write_gambit_neutral(fn,nodes,elems)
    elif ftype == 'smesh':
        write_smesh(fn,nodes,elems)
    elif ftype == 'gts':
        edges,faces = compactEdges(elems)
        write_gts(fn,nodes,edges,faces)
    else:
        print "Cannot save as file %s" % fn
   

def stl_to_off(stlname,offname=None,sanitize=True):
    """Transform an .stl file to .off format."""
    if not offname:
        offname = changeExt(stlname,'.off')
    if sanitize:
        options = ''
    else:
        # admesh always wants to perform some actions on the STL. The -c flag
        # to suppress all actions makes admesh hang. Therefore we include the
        # action -d (fix normal directions) as the default.
        options = '-d'    
    runCommand("admesh %s --write-off '%s' '%s'" % (options,offname,stlname))
    return offname


def stl_to_gts(stlname,outname=None):
    """Transform an .stl file to .gts format."""
    if not outname:
        outname = changeExt(stlname,'.gts')
    runCommand("stl2gts < '%s' > '%s'" % (stlname,outname))
    return outname


def stl_to_femodel(formex,sanitize=True):
    """Transform an .stl model to FEM model.

    This is a faster alternative for the Formex feModel() method.
    It works by writing the model to file, then using admesh to convert
    the .stl file to .off format, and finally reading back the .off.

    Returns a tuple of (nodes,elems). If sanitize is False, the result will be
    such that Formex(nodes[elems]) == formex. By default, admesh sanitizes the
    STL model and may remove/fix some elements.
    """
    fn = changeExt(os.path.tempnam('.','pyformex-tmp'),'.stl')
    write_ascii(fn,formex.f)
    return read_stl(fn,sanitize)


def off_to_tet(fn):
    """Transform an .off model to tetgen (.node/.smesh) format."""
    GD.message("Transforming .OFF model %s to tetgen .smesh" % fn)
    nodes,elems = read_off(fn)
    write_node_smesh(changeExt(fn,'.smesh'),nodes,elems)

   
def compactEdges(elems):
    """Transform elems to an edges,faces tuple.

    elems is an (nelems,nplex) int32 array of element node numbers.

    Returns a tuple edges,faces where
    edges is an (nedges,2) int32 array of edges connecting two node numbers.
    faces is an (nelems,nplex) int32 array with the edge numbers connecting
    each pair os subsequent nodes in the elements of elems.

    The order of the edges respects the node order, and starts with nodes 0-1.
    The node numbering in the edges is always lowest node number first.
    """
    nelems,nplex = elems.shape
    magic = elems.max() + 1
    if magic > 2**31:
        raise RuntimeError,"Cannot compact edges for more than 2**31 nodes"
    n = arange(nplex)
    edg = column_stack([n,roll(n,1)])
    alledges = elems[:,edg]
    # sort edge nodes with lowest number first
    alledges.sort()
    edg = alledges.astype(int64).reshape((-1,2))
    # encode the two node numbers in a single edge number
    codes = edg[:,0] * magic + edg[:,1]
    # keep the unique edge numbers
    uniqid,uniq = unique1d(codes,True)
    # we suppose uniq is sorted 
    uedges = uniq.searchsorted(codes)
    edges = column_stack([uniq/magic,uniq%magic])
    faces = uedges.reshape((nelems,nplex))
    return edges,faces


def expandEdges(edges,faces):
    elems = edges[faces]
    flag0 = (elems[:,0,0]==elems[:,1,0]) + (elems[:,0,0]==elems[:,1,1])
    flag2 = (elems[:,2,0]==elems[:,1,0]) + (elems[:,2,0]==elems[:,1,1])
    nod0 = where(flag0,elems[:,0,1],elems[:,0,0])
    nod1 = where(flag0,elems[:,0,0],elems[:,0,1])
    nod2 = where(flag2,elems[:,2,0],elems[:,2,1])
    elems = column_stack([nod1,nod2,nod0])
    return elems


def border(elems):
    """Detect the border elements of an STL model.

    The input is an (nelems,3) integer array of elements each defined
    by 3 node numbers.
    The return value is an (nelems,3) bool array flagging all the border edges.
    The result can be further used as follows:
      where(result) gives a tuple of indices of the border edges
      result.sum(axis=1) gives the number of border edges for all elements
      where(any(result,axis=1))[0] gives a list of elements with borders
    """
    magic = elems.max() + 1
    if magic > 2**31:
        raise RuntimeError,"Cannot detect border for more than 2**31 nodes"

    triedges = [ [0,1], [1,2], [2,0] ]
    # all the edges of all elements (nelems,3,2)
    edges = elems[:,triedges].astype(int64)
    # encode the edges and reverse edges
    codes = edges[:,:,0] * magic + edges[:,:,1]  
    rcodes = edges[:,:,1] * magic + edges[:,:,0]
    # sort the edges to facilitate searching
    fcodes = codes.ravel()
    fcodes.sort()
    # lookup reverse edges matching edges: if they exist, fcodes[pos]
    # will equal rcodes
    pos = fcodes.searchsorted(rcodes).clip(min=0,max=fcodes.shape[0]-1)
    return fcodes[pos] != rcodes


def nborder(elems):
    """Detect the border elements of an STL model.

    Returns an (nelems) integer array with the number of border edges
    for each all elements. This is equivalent to
    border(elems).sum(axis=1).
    """
    return border(elems).sum(axis=1)


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
    print elems,remove
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
    
