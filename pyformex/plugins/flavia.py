#!/usr/bin/env pyformex
# $Id$

"""
Interface with flavia FE result files.

(C) 2010 Benedict Verhegghe.
"""
from arraytools import *
from plugins.mesh import Mesh
from plugins.fe_post import FeResult
import shlex

element_type_translation = {
    'quadrilateral': {
        4: 'quad4',
        8: 'quad8',
        9: 'quad9',
        },
    }
element_results_count = {
    'vector': {
        2: 2,
        3: 3,
        },
    'matrix': {
        2: 3,
        3: 6,
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
            s = shlex.split(line.lower())
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
    M = Mesh(coords,elems,props).compact()
    M.ndim = ndim
    return M
            
            

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
                props = growAxis(props,nelems,axis=0,fill=0)
                nelems = elems.shape[0]
            elems[i-1] = e
            props[i-1] = p
    defined = elems.sum(axis=1) > 0
    return elems[defined]-1,props[defined]


def readResults(fn,mesh):
    """Read a flavia results file for an ndim mesh.

    """
    fil = open(fn,'r')

    results = {}
    for line in fil:
        if line.startswith('#'):
            continue
        elif line.startswith('Result'):
            s = shlex.split(line.lower())
            name = s[1]
            restype = s[4]
            domain = s[5]
            print domain
            if domain != 'onnodes':
                print "Currently only results on nodes can be read"
                print "Skipping %s %s" % (name,domain)
                nres = 0
                continue
            nres = element_results_count[restype][mesh.ndim]
        elif line.startswith('Values'):
            if nres > 0:
                result = readResult(fil,mesh.nnodes(),nres)
                print name
                results[name] = result
        else:
            print line
    return results


def readResult(fil,nvalues,nres):
    """Read a set of results from a flavia file"""
    values = zeros((nvalues,nres),dtype=Float)
    for line in fil:
        if line.startswith('End Values'):
            break
        else:
            s = line.split()
            i = int(s[0])
            x = map(float,s[1:])
            values[i-1] = x
    return values

from plugins.fe import Model
from plugins.fe_post import FeResult

def createFeResult(mesh,results):
    
    model = Model(mesh.coords,mesh.elems)
    DB = FeResult()
    DB.nodes = model.coords
    DB.nnodes = model.coords.shape[0]
    DB.nodid = arange(DB.nnodes)
    DB.elems = dict(enumerate(model.elems))
    DB.nelems = model.celems[-1]
    DB.Finalize()

    ndisp = results['displacement'].shape[1]
    nstrs = results['stress'].shape[1]
    DB.datasize['U'] = ndisp
    DB.datasize['S'] = nstrs
    for lc in range(1):  # currently only 1 step
        DB.Increment(lc,0)
        DB.R['U'] = results['displacement']
        DB.R['S'] = results['stress']
    return DB


def readFlavia(meshfile,resfile):
    """Read flavia results files"""
    M = readMesh(meshfile)
    R = readResults(resfile,M)
    DB = createFeResult(M,R)
    DB.printSteps()
    return DB
   

if __name__ == "draw":
    chdir('/home/bene/prj/pyformex')
    name = 'FeResult-001'
    meshfile = name+'.flavia.msh'
    resfile = utils.changeExt(meshfile,'res')
    M = readMesh(meshfile)
    print M.coords.shape,M.elems.shape
    print M.coords,M.elems
    draw(M)
    R = readResults(resfile,M)
    DB = createFeResult(M,R)
    DB.printSteps()
    print DB.R
    print DB.datasize
    DB1 = FeResult()
    print DB1.datasize

    for key in [ 'U0','U1','U2','U3']:
        v = DB.getres(key)
        if v is not None:
            print "%s: %s" % (key,v.shape)

    

    
# End

