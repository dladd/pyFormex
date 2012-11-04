# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""
Interface with flavia FE result files.

(C) 2010 Benedict Verhegghe.
"""
from __future__ import print_function
from arraytools import *
from mesh import Mesh,mergeMeshes
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

    Returns a list of Meshes if succesful.
    """
    fil = open(fn,'r')

    meshes = []
    for line in fil:
        if line.startswith('#'):
            continue
        elif line.startswith('Mesh'):
            s = shlex.split(line.lower())
            s = dict(zip(s[0::2],s[1::2]))
            print(s)
            ndim = int(s['dimension'])
            nplex = int(s['nnode'])
            eltype = element_type_translation[s['elemtype']][nplex]
            print("eltype = %s, ndim = %s" % (eltype,ndim))
        elif line.startswith('Coordinates'):
            coords = readCoords(fil,ndim)
            print("Coords %s" % str(coords.shape))
        elif line.startswith('Elements'):
            elems,props = readElems(fil,nplex)
            print("Elements %s %s" % (elems.shape,props.shape))
            meshes.append((elems,props))
        else:
            print(line)
    elems,props = [m[0] for m in meshes],[m[1] for m in meshes]
    maxnod = max([ e.max() for e in elems])
    coords = coords[:maxnod+1]
    return coords,elems,props,ndim
            

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


def readResults(fn,nnodes,ndim):
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
            print(domain)
            if domain != 'onnodes':
                print("Currently only results on nodes can be read")
                print("Skipping %s %s" % (name,domain))
                nres = 0
                continue
            nres = element_results_count[restype][ndim]
        elif line.startswith('Values'):
            if nres > 0:
                result = readResult(fil,nnodes,nres)
                print(name)
                results[name] = result
        else:
            print(line)
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


def createFeResult(model,results):
    """Create an FeResult from meshes and results"""
    #model = Model(*mergeMeshes(meshes,fuse=False))
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
    """Read flavia results files

    Currently we only read matching pairs of meshfile,resfile files.
    """
    if meshfile:
        coords,elems,props,ndim = readMesh(meshfile)
    if resfile:
        R = readResults(resfile,coords.shape[0],ndim)
    M = Model(coords,elems)
    DB = createFeResult(M,R)
    DB.printSteps()
    return DB
   

if __name__ == "draw":
    chdir('/home/bene/prj/pyformex')
    name = 'FeResult-001'
    meshfile = name+'.flavia.msh'
    resfile = utils.changeExt(meshfile,'res')
    M = readMesh(meshfile)
    print(M.coords.shape,M.elems.shape)
    print(M.coords,M.elems)
    draw(M)
    R = readResults(resfile,M)
    DB = createFeResult(M,R)
    DB.printSteps()
    print(DB.R)
    print(DB.datasize)
    DB1 = FeResult()
    print(DB1.datasize)

    for key in [ 'U0','U1','U2','U3']:
        v = DB.getres(key)
        if v is not None:
            print("%s: %s" % (key,v.shape))

    

    
# End

