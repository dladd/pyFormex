#!/usr/bin/env pyformex
# $Id$
#
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import globaldata as GD
from utils import runCommand, changeExt
from numpy import *
from formex import *


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


def read_ascii(f,dtype=float32,large=False):
    """Read an ascii .stl file into an [n,3,3] float array.

    If the .stl is large, read_ascii_large() is recommended, as it is
    a lot faster.
    """
    if large:
        return read_ascii_large(f,dtype=dtype)
    own = type(f) == str
    if own:
        f = file(f,'r')
    n = 100
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
        
    if own:
        f.close()
    if finished:
        return a[:i]
    raise RuntimeError,"Incorrect stl file: read %d lines, %d facets" % (cnt,i)
        


def read_ascii_large(f,dtype=float32):
    """Read an ascii .stl file into an [n,3,3] float array.

    This is an alternative for read_ascii, which is a lot faster on large
    STL models.
    It requires the 'awk' command though, so is probably only useful on
    Linux/UNIX. It works by first transforming  the input file to a
    .nodes file and then reading it through numpy's fromfile() function.
    """
    tmp = '%s.nodes' % f
    import timer
    t = timer.Timer()
    runCommand("awk '/^[ ]*vertex[ ]+/{print $2,$3,$4}' %s | d2u > %s" % (f,tmp))
    print "Converting phase: %s seconds" % t.seconds()
    t.reset()
    nodes = fromfile(tmp,sep=' ',dtype=dtype).reshape((-1,3,3))
    print "Input phase: %s seconds" % t.seconds()
    return nodes


def read_off(fn):
    """Read an .off surface mesh.

    Returns a nodes,elems tuple.
    """
    print "Reading .OFF %s" % fn
    fil = file(fn,'r')
    mode = -1
    for line in fil:
        if mode < 0:
            if line.startswith('OFF'):
                mode = 0
            else:
                raise RuntimeError,"File %s does not seem to be an .OFF file" % fn
        elif mode == 0:
            nnodes,nelems,nedges = map(int,line.split())
            nodes = zeros((nnodes,3),dtype=Float)
            elems = zeros((nelems,3),dtype=Int)
            count = 0
            mode = 1
        elif mode == 1:
            nodes[count] = map(float,line.split()[:3])
            count += 1
            if count >= nnodes:
                count = 0
                mode = 2
        elif mode == 2:
            elems[count] = map(int,line.split()[1:4]) # each line has a '3'
            count += 1
            if count >= nelems:
                print "Read %s nodes and %s faces" % (nnodes,nelems)
                return nodes,elems
    raise RuntimeError,"OFF format read erro on file %s" % fn


def stl_to_off(stlname,offname=None,sanitize=True):
    """Transform an .stl model to .off format."""
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


def off_to_tet(fn):
    """Transform an .off model to tetgen (.node/.smesh) format."""
    message("Transforming .OFF model %s to tetgen .smesh" % fn)
    nodes,elems = read_off(fn)
    write_node_smesh(changeExt(fn,'.smesh'),nodes,elems)

    

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print a
    
