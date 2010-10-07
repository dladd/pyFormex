#!/usr/bin/env pyformex
# $Id$

"""
Interface with flavia FE result files.

(C) 2010 Benedict Verhegghe.
"""

element_type_translation = {
    'quadrilateral': {
        4: 'quad4',
        8: 'quad8',
        9: 'quad9',
        },
    }

######################### functions #############################


def readMesh(fn):
    """Read a flavia mesh file.

    Returns a Mesh if succesful.
    """
    fil = open(fn,'r')

    for line in fil:
        if line.startswith('#'):
            continue
        elif line.startswith('Mesh'):
            s = line.lower().split()
            s = dict(zip(s[0::2],s[1::2]))
            print s
            ndim = int(s['dimension'])
            nplex = int(s['nnode'])
            eltype = element_type_translation[s['elemtype']][nplex]
            print "eltype = %s, ndim = %s" % (eltype,ndim)
        elif line.startswith('Coordinates'):
            coords = readCoords(fil,ndim)
            print coords.shape
        elif line.startswith('Elements'):
            elems,props = readElems(fil,nplex)
            print elems.shape
            print props.shape
        else:
            print line
    return Mesh(coords,elems,props).compact()
            
            

def readCoords(fil,ndim):
    """Read a set of coordinates from a flavia file"""
    ncoords = 100
    coords = zeros((ncoords,3),dtype=Float)
    for line in fil:
        if line.startswith('End Coordinates'):
            break
        else:
            s = line.split()
            i = int(s[0])
            x = map(float,s[1:])
            while i >= ncoords:
                coords = growAxis(coords,ncoords,axis=0,fill=0.0)
                ncoords = coords.shape[0]
            coords[i-1,:ndim] = x
    return coords       


def readElems(fil,nplex):
    """Read a set of coordinates from a flavia file"""
    nelems = 100
    elems = zeros((nelems,nplex),dtype=Int)
    props = zeros((nelems),dtype=Int)
    for line in fil:
        if line.startswith('End Elements'):
            break
        else:
            s = line.split()
            i = int(s[0])
            e = map(int,s[1:nplex+1])
            p = int(s[nplex+1])
            while i >= nelems:
                elems = growAxis(elems,nelems,axis=0,fill=0)
                props = growAxis(pros,nelems,axis=0,fill=0)
                nelems = elems.shape[0]
            elems[i-1] = e
            props[i-1] = p
    defined = elems.sum(axis=1) > 0
    return elems[defined]-1,props[defined]


def readResults(fn):
    """Read a flavia results file.

    """
    fil = open(fn,'r')

    for line in fil:
        if line.startswith('#'):
            continue
        elif line.startswith('Mesh'):
            s = line.lower().split()
            s = dict(zip(s[0::2],s[1::2]))
            print s
            ndim = int(s['dimension'])
            nplex = int(s['nnode'])
            eltype = element_type_translation[s['elemtype']][nplex]
            print "eltype = %s, ndim = %s" % (eltype,ndim)
        elif line.startswith('Coordinates'):
            coords = readCoords(fil,ndim)
            print coords.shape
        elif line.startswith('Elements'):
            elems,props = readElems(fil,nplex)
            print elems.shape
            print props.shape
        else:
            print line
    return Mesh(coords,elems,props).compact()
            


if __name__ == "draw":
    chdir('/home/bene/prj/pyformex')
    meshfile = 'FeResult-001.flavia.msh'
    resfile = utils.changeExt(meshfile,'res')
    M = readMesh(meshfile)
    print M.coords.shape,M.elems.shape
    print M.coords,M.elems
    draw(M)
    
# End

