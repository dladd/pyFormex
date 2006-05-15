#!/usr/bin/env pyformex
# $Id$
#
"""Import/Export Formex structures to/from stl format.

An stl is stored as a numerical array with shape [n,3,3].
This is compatible with the pyFormex data model.
"""

import globaldata as GD
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


def stl_import_ascii(f):
    """Read an ascii .stl file into an [n,3,3] float array"""

    own = type(f) == str
    if own:
        f = file(f,'r')
    n = 100
    a = zeros(shape=[n,3,3],dtype=float32)
    x = zeros(shape=[3,3],dtype=float32)
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
        

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print a
    
