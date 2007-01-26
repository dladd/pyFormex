#!/usr/bin/env pyformex
# $Id$
#
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import globaldata as GD
from plugins import tetgen
from utils import runCommand, changeExt,countLines
from numpy import *
from formex import Formex
import os


# The Stl class should not be used yet! Use the functions instead.
class Stl(Formex):
    """A specialized Formex subclass representing stl models."""

    def __init__(self,*args):
        Formex.__init__(self,*args)
        if self.f.shape[1] != 3:
            if self.f.size == 0:
                self.f.shape = (0,3,3)
            else:
                raise RuntimeError,"Stl: should have 3 nodes per element"   
        print self.f.shape
        self.n = None
        self.a = None


def compute_normals(a,normalized=True):
    """Compute the normal vectors of the triangles in a[n,3,3].

    Default is to normalize the vectors to unit length.
    If not essential, this can be switched off to save computing time.
    """
    n = cross(a[:,1]-a[:,0],a[:,2]-a[:,1])
    if normalized:
        n /= column_stack([sqrt(sum(n*n,-1))])
    return n


def write_ascii(a,f):
    """Export an [n,3,3] float array as an ascii .stl file."""

    own = type(f) == str
    if own:
        f = file(f,'w')
    f.write("solid  Created by %s\n" % GD.Version)
    v = compute_normals(a)
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


def write_stl(fn,nodes,elems):
    write_ascii(Formex(nodes[elems]),fn)

def write_off(fn,nodes,elems):
    if nodes.shape[1] != 3 or elems.shape[1] != 3:
        raise runtimeError, "Invalid arguments or shape"
    fil = file(fn,'w')
    fil.write("OFF\n")
    fil.write("%s %s 0\n" % (nodes.shape[0],elems.shape[0]))
    fil.write(str(nodes)[1:-1])
    fil.close()

def write_neu(fn,nodes,elems):
    pass
def write_smesh(fn,nodes,elems):
    tetgen.write_node_smesh(fn,nodes,elems)

def saveSurface(nodes,elems,fn):
    ext = os.path.splitext(fn)[1]
    print "Saving as type %s" % ext
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
    write_ascii(formex.f,fn)
    return read_stl(fn,sanitize)


def off_to_tet(fn):
    """Transform an .off model to tetgen (.node/.smesh) format."""
    message("Transforming .OFF model %s to tetgen .smesh" % fn)
    nodes,elems = read_off(fn)
    write_node_smesh(changeExt(fn,'.smesh'),nodes,elems)



def border(elems):
    """Detect the border elements of an STL model.

    The input is an (nelems,3) integer array of elements each defined
    by 3 node numbers.
    The return value is an (nelems) array holding the number (0,1,2 or 3)
    of border edges for each element.
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
    # lookup reverse edges matching edges
    pos = fcodes.searchsorted(rcodes)
    f = (fcodes[pos] != rcodes).astype(int32)
    return f.sum(axis=1)
    

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print a
    
