#!/usr/bin/env pyformex
# $Id$
#
"""Import/Export Formex structures to/from stl format"""

import globaldata as GD
from numpy import *


def write_ascii(a,f):
    """Export an [n,3,3] float array as an ascii .stl file"""

    f.write("solid  Created by %s\n" % GD.Version)
    for e in F:
        normal = numpy.cross(e[1]-e[0],e[2]-e[1])
        f.write("  facet normal %f %f %f\n" % tuple(normal))
        f.write("    outer loop\n")
        for p in e:
            f.write("      vertex %f %f %f\n" % tuple(p))
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write("endsolid\n")


def read_error(cnt,line):
    """Raise an error on reading the stl file."""
    raise RuntimeError,"Invalid .stl format while reading line %s\n%s" % (cnt,line)


def read_ascii(f):
    """Read an ascii .stl file into an [n,3,3] float array"""
    n = 100
    a = zeros(shape=[n,3,3],dtype=float32)
    x = zeros(shape=[3,3],dtype=float32)
    i = 0
    j = 0
    cnt = 0
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
            return a[:i]

if __name__ == '__main__':
    f = file('unit_triangle.stl','r')
    a = read_ascii(f)
    f.close()
    print a
    
