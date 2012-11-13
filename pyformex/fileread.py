# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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
"""Read geometry from file in a whole number of formats.

This module defines basic routines to read geometrical data
from a file and the specialized importers to read files in a number of
well known standardized formats.

The basic routines are very versatile as well as optimized (using the version
in the pyFormex C-library) and allow to easily create new exporters for
other formats.
"""
from __future__ import print_function

import pyformex as pf
from mesh import *
import utils
from lib import misc
import os


def getParams(line):
    """Strip the parameters from a comment line"""
    s = line.split()
    d = {'mode': s.pop(0),'filename': s.pop(0)}
    d.update(dict(zip(s[::2],s[1::2])))
    return d


def readNodes(fil):
    """Read a set of nodes from an open mesh file"""
    a = fromfile(fil,sep=" ").reshape(-1,3)
    x = Coords(a)
    return x


def readElems(fil,nplex):
    """Read a set of elems of plexitude nplex from an open mesh file"""
    print("Reading elements of plexitude %s" % nplex)
    e = fromfile(fil,sep=" ",dtype=Int).reshape(-1,nplex) 
    e = Connectivity(e)
    return e


def readEsets(fil):
    """Read the eset data of type generate"""
    data = []
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 4:
            data.append(s[:1]+map(int,s[1:]))
    return data
            

def readMeshFile(fn):
    """Read a nodes/elems model from file.

    Returns a dict:

    - `coords`: a Coords with all nodes
    - `elems`: a list of Connectivities
    - `esets`: a list of element sets
    """
    d = {}
    fil = open(fn,'r')
    for line in fil:
        if line[0] == '#':
            line = line[1:]
        globals().update(getParams(line))
        dfil = open(filename,'r')
        if mode == 'nodes':
            d['coords'] = readNodes(dfil)
        elif mode == 'elems':
            elems = d.setdefault('elems',[])
            e = readElems(dfil,int(nplex)) - int(offset)
            elems.append(e)
        elif mode == 'esets':
            d['esets'] = readEsets(dfil)
        else:
            print("Skipping unrecognized line: %s" % line)
        dfil.close()

    fil.close()
    return d                    


def extractMeshes(d):
    """Extract the Meshes read from a .mesh file.

    """
    x = d['coords']
    e = d['elems']
    M = [ Mesh(x,ei) for ei in e ]
    return M


def convertInp(fn):
    """Convert an Abaqus .inp to a .mesh set of files"""
    converter = os.path.join(pf.cfg['pyformexdir'],'bin','read_abq_inp.awk')
    fn = os.path.abspath(fn)
    dirname = os.path.dirname(fn)
    basename = os.path.basename(fn)
    cmd = 'cd %s;%s %s' % (dirname,converter,basename)
    sta,out = utils.runCommand(cmd)


def readInpFile(filename):
    """Read the geometry from an Abaqus/Calculix .inp file

    This is a replacement for the convertInp/readMeshFile combination.
    It uses the ccxinp plugin to provide a direct import of the Finite
    Element meshes from an Abaqus or Calculix input file.
    Currently still experimental and limited in functionality (aimed
    primarily at Calculix). But also many simple meshes from Abaqus can
    already be read.

    Returns an fe.Model instance.
    """
    from plugins import ccxinp,fe
    ccxinp.skip_unknown_eltype = True
    model = ccxinp.readInput(filename)
    print("Number of parts: %s" % len(model.parts))
    fem = {}
    for part in model.parts:
        try:
            coords = Coords(part['coords'])
            nodid = part['nodid']
            nodpos = inverseUniqueIndex(nodid)
            print("nnodes = %s" % coords.shape[0])
            print("nodid: %s" % nodid)
            print("nodpos: %s" % nodpos)
            for e in part['elems']:
                print("Orig els %s" % e[1])
                print("Trl els %s" % nodpos[e[1]])
            elems = [ Connectivity(nodpos[e],eltype=t) for (t,e) in part['elems'] ]
            fem[part['name']] = fe.Model(coords,elems)
        except:
            print("Skipping part %s" % part['name'])
    return fem
    

def read_gambit_neutral(fn):
    """Read a triangular surface mesh in Gambit neutral format.

    The .neu file nodes are numbered from 1!
    Returns a nodes,elems tuple.
    """
    scr = os.path.join(pf.cfg['bindir'],'gambit-neu ')
    utils.runCommand("%s '%s'" % (scr,fn))
    nodesf = utils.changeExt(fn,'.nodes')
    elemsf = utils.changeExt(fn,'.elems')
    nodes = fromfile(nodesf,sep=' ',dtype=Float).reshape((-1,3))
    elems = fromfile(elemsf,sep=' ',dtype=int32).reshape((-1,3))
    return nodes, elems-1


def read_gambit_neutral_hex(fn):
    """Read an hexahedral mesh in Gambit neutral format.

    The .neu file nodes are numbered from 1!
    Returns a nodes,elems tuple.
    """
    scr = os.path.join(pf.cfg['bindir'],'gambit-neu-hex ')
    print("%s '%s'" % (scr,fn))
    utils.runCommand("%s '%s'" % (scr,fn))
    nodesf = utils.changeExt(fn,'.nodes')
    elemsf = utils.changeExt(fn,'.elems')
    nodes = fromfile(nodesf,sep=' ',dtype=Float).reshape((-1,3))
    elems = fromfile(fn_e,sep=' ',dtype=int32).reshape((-1,8))
    elems = elems[:,(0,1,3,2,4,5,7,6)]
    return nodes, elems-1

# End

