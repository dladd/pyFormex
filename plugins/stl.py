#!/usr/bin/env pyformex
# $Id$
#
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import globaldata as GD
from plugins import tetgen
from utils import runCommand, changeExt,countLines,mtime
from numpy import *
from formex import Formex
import os


# The Stl class should not be used yet! Use the functions instead.
class Stl(object):
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
        pass


def compute_normals(a,normalized=True):
    """Compute the normal vectors of the triangles in a[n,3,3].

    Default is to normalize the vectors to unit length.
    If not essential, this can be switched off to save computing time.
    """
    n = cross(a[:,1]-a[:,0],a[:,2]-a[:,1])
    if normalized:
        n /= column_stack([sqrt(sum(n*n,-1))])
    return n


def write_ascii(f,a):
    """Export an [n,3,3] float array as an ascii .stl file."""

    own = type(f) == str
    if own:
        f = file(f,'w')
    f.write("solid  Created by %s\n" % GD.Version)
    v = compute_normals(a)
    print "Degenerate facets",where(isnan(v))
    for e,n in zip(a,v):
        f.write("  facet normal %f %f %f\n" % tuple(n))
        f.write("    outer loop\n")
        for p in e:
            f.write("      vertex %f %f %f\n" % tuple(p))
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write("endsolid\n")
    if own:
        f.close()


def read_error(cnt,line):
    """Raise an error on reading the stl file."""
    raise RuntimeError,"Invalid .stl format while reading line %s\n%s" % (cnt,line)


def read_ascii(fn,dtype=float32,large=False,guess=False,off=False):
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
        


def read_ascii_large(fn,dtype=float32):
    """Read an ascii .stl file into an [n,3,3] float array.

    This is an alternative for read_ascii, which is a lot faster on large
    STL models.
    It requires the 'awk' command though, so is probably only useful on
    Linux/UNIX. It works by first transforming  the input file to a
    .nodes file and then reading it through numpy's fromfile() function.
    """
    tmp = '%s.nodes' % fn
    runCommand("awk '/^[ ]*vertex[ ]+/{print $2,$3,$4}' %s | d2u > %s" % (fn,tmp))
    nodes = fromfile(tmp,sep=' ',dtype=dtype).reshape((-1,3,3))
    return nodes


def read_stl(fn,sanitize=False):
    """Read an .stl file into an (nodes,elems) femodel.

    This is done by first coverting the .stl to .off format.
    """
    offname = stl_to_off(fn,sanitize=False)
    return read_off(offname)


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
    nodes = fromfile(file=fil, dtype=float32, count=3*nnodes, sep=' ')
    # elems have number of vertices + 3 vertex numbers
    elems = fromfile(file=fil, dtype=int32, count=4*nelems, sep=' ')
    print "Read %d nodes and %d elems" % (nnodes,nelems)
    return nodes.reshape((-1,3)),elems.reshape((-1,4))[:,1:]


def read_gambit_neutral(fn):
    """Read a triangular surface mesh in Gambit neutral format.

    The .neu file nodes are numbered from 1!
    Returns a nodes,elems tuple.
    """
    runCommand("%s/external/gambit-neu %s" % (GD.cfg['pyformexdir'],fn))
    nodesf = changeExt(fn,'.nodes')
    elemsf = changeExt(fn,'.elems')
    nodes = fromfile(nodesf,sep=' ',dtype=float32).reshape((-1,3))
    elems = fromfile(elemsf,sep=' ',dtype=int32).reshape((-1,3))
    return nodes, elems-1

def readSurface(fn):
    ext = os.path.splitext(fn)[1]
    if ext == '.stl':
        ofn = changeExt(fn,'.off')
        if os.path.exists(ofn) and mtime(ofn) > mtime(fn):
            # There is a more recent .off, let's use it
            nodes,elems = read_off(ofn)
        else:
            nodes,elems = read_stl(fn)
    elif ext == '.off':
        nodes,elems = read_off(fn)
    elif ext == '.neu':
        nodes,elems = read_gambit_neutral(fn)
    elif ext == '.smesh':
        nodes,elems = tetgen.readSurface(fn)
    else:
        print "Cannot read file %s" % fn
    return nodes,elems


def write_stl(fn,nodes,elems):
    write_ascii(fn,nodes[elems])

def write_off(fn,nodes,elems):
    if nodes.shape[1] != 3 or elems.shape[1] != 3:
        raise runtimeError, "Invalid arguments or shape"
    fil = file(fn,'w')
    fil.write("OFF\n")
    fil.write("%s %s 0\n" % (nodes.shape[0],elems.shape[0]))
    for nod in nodes:
        fil.write("%s %s %s\n" % tuple(nod))
    for el in elems:
        fil.write("%s %s %s\n" % tuple(el))
    fil.close()

def write_neu(fn,nodes,elems):
    pass
def write_smesh(fn,nodes,elems):
    tetgen.writeSurface(fn,nodes,elems)

def writeSurface(fn,nodes,elems):
    ext = os.path.splitext(fn)[1]
    print "Writing %s triangles to file %s" % (elems.shape[0],fn)
    if ext == '.stl':
        write_stl(fn,nodes,elems)
    elif ext == '.off':
        write_off(fn,nodes,elems)
    elif ext == '.neu':
        write_neu(fn,nodes,elems)
    elif ext == '.smesh':
        write_smesh(fn,nodes,elems)
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
    runCommand("admesh %s --write-off %s %s" % (options,offname,stlname))
    return offname


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
        print "TOO MANY ELEMENTS: can not determine border edges"
        return elems

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
    
