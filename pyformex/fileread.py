# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

    - `node`: a Coords with all nodes
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
    sta, out = utils.runCommand(cmd)
    

# End

