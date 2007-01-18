#!/usr/bin/env python
# $Id$
"""read/write tetgen format files."""

from numpy import *

def invalid(line,fn):
    """Print message for invalid line."""
    print "The following line in file %s is invalid:" % fn
    print line
    

def readNodes(fn,offset=-1):
    """Read a tetgen .node file. Returns an array of points."""
    fil = file(fn,'r')
    nodes = None
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 0 or s[0][0] == '#':
            continue
        if nodes is None:
            try:
                npts,ndim,nattr,nbmark = map(int,s)
            except:
                invalid(line,fn)
                raise
            nodes = zeros((npts,ndim),float)
        else:
            i = int(s[0])
##             if i < 0 or i >npts: 
##                 invalid(line,fn)
##                 raise RuntimeError
            try:
                nodes[i+offset] = map(float,s[1:ndim+1])
            except:
                invalid(line,fn)
                raise
    return nodes


def readElems(fn):
    """Read a tetgen .ele file. Returns an array of tetraeder elements."""
    fil = file(fn,'r')
    elems = None
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 0 or s[0][0] == '#':
            continue
        if elems is None:
            try:
                nels,nnod,nattr = map(int,s)
            except:
                invalid(line,fn)
                raise
            elems = zeros((nels,nnod),int)
        else:
            i = int(s[0])
            if i < 1 or i >nels: 
                invalid(line,fn)
                raise RuntimeError
            try:
                elems[i-1] = map(int,s[1:nnod+1])
            except:
                invalid(line,fn)
                raise
    return elems


def readSurface(fn):
    """Read a tetgen .smesh file. Returns an array of triangle elements."""
    fil = file(fn,'r')
    part = 0
    elems = None
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 0:
            continue
        if s[0][0] == '#':
            if len(s) >= 3 and s[1] == 'part':
                part = int(s[2][0])
            continue
        if part == 2:
            if elems is None:
                try:
                    nels = int(s[0])
                    nnod = 3
                except:
                    invalid(line,fn)
                    raise
                if nels > 0:
                    elems = zeros((nels,nnod),int)
                    i = 0
                else:
                    invalid(line,fn)
                    raise RuntimeError
            else:
                n = int(s[0])
                if n == nnod:
                    try:
                        elems[i] = map(int,s[1:nnod+1])
                        i += 1
                    except:
                        invalid(line,fn)
                        raise
                else:
                    invalid(line,fn)
                    raise RuntimeError                 
    return elems


def writeNodes(fn,nodes):
    """Write a tetgen .node file."""
    fil = file(fn,'w')
    fil.write("%s %s 0 0" % nodes.shape)
    for i,n in enumerate(nodes):
        fil.write("%s %s %s %s" % (i,n[0],n[1],n[2]))
    fil.close()


if __name__ == "__main__":
    import sys

    for f in sys.argv[1:]:
        if f.endswith('.node'):
            nodes = readNodes(f)
            print "Read %d nodes" % nodes.shape[0]
        elif f.endswith('.ele'):
            elems = readElems(f)
            print "Read %d elems" % elems.shape[0]
        elif f.endswith('.smesh'):
            elems = readSurface(f)
            print "Read %d triangles" % elems.shape[0]
        
    
