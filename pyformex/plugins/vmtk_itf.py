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
"""Operations on triangulated surfaces using VMTK functions.

This module provides access to VMTK functionality from inside pyFormex.
Documentation for VMTK can be found on http://www.vmtk.org/Main/Tutorials/
and http://www.vmtk.org/VmtkScripts/vmtkscripts/
"""
from __future__ import print_function

from plugins.trisurface import TriSurface
import pyformex as pf
import utils
import os

## from plugins.trisurface import *
## from utils import runCommand
## chdir(__file__)


## clear()


## examplefile = os.path.join(getcfg('datadir'),'bifurcation.off')
## print(examplefile)
## s= TriSurface.read(examplefile)

## draw(s)

## dir=''
## fnin = dir+'tempi.stl'

###################################################################
#############centerline example####################################
#def readVmtkCenterlineDat(fn):
#    """Read a file .dat containing the centerlines generated with vmtk.
#
#    The first line may contain a header.
#    All other lines ('nlines') contain  'nf' floats.
#    All data are seperated by blanks.
#    
#    The return value is a tuple of
#    - a float array (nlines,nf) with the data
#    - a list with the identifiers from the first line
#    """
#    fil = file(fn,'r')
#    line = fil.readline()
#    s = line.strip('\n').split()
#    data = fromfile(fil,sep=' ',dtype=float32)
#    data = data.reshape((-1,len(s)))
#    return data, s
#
#fnout = dir+'cl.dat'
#cmd = 'vmtk vmtkcenterlines -ifile %s -ofile %s'%(fnin, fnout)
#sta,out=runCommand(cmd)
#print((sta,out))
#data, header = readVmtkCenterlineDat(fnout)
#clp = data[:, :3]
#print (header)
#draw(Coords(clp))
#exit()
###################################################################
###################################################################


def remesh(self,edgelen):
    """Remesh a TriSurface.

    edgelen is the suggested edge length
    """
    tmp = utils.tempFile(suffix='.stl').name
    tmp1 = utils.tempFile(suffix='.stl').name
    pf.message("Writing temp file %s" % tmp)
    self.write(tmp,'stl')
    pf.message("Remeshing using VMTK")
    cmd = "vmtk vmtksurfaceremeshing -ifile %s -ofile %s -edgelength %s" % (tmp,tmp1,edgelen)
    sta,out = utils.runCommand(cmd)
    os.remove(tmp)
    if sta:
        pf.message("An error occurred during the remeshing.")
        pf.message(out)
        return None
    S = TriSurface.read(tmp1)
    os.remove(tmp1)
    return S


def install_trisurface_methods():
    """Install extra TriSurface methods

    """
    from plugins.trisurface import TriSurface
    print("INSTALLING VMTK METHODS:1")
    TriSurface.remesh = remesh

install_trisurface_methods()

# End













