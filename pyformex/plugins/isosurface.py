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
#
"""Isosurface: surface reconstruction algorithms

This module contains the marching cube algorithm.

Some of the code is based on the example by Paul Bourke from
http://paulbourke.net/geometry/polygonise/
   
"""
from __future__ import print_function

import numpy as np
from multi import multitask,cpu_count,splitar


def isosurface(data,level,nproc=-1):
    """Create an isosurface through data at given level.

    - `data`: (nx,ny,nz) shaped array of data values at points with
      coordinates equal to their indices. This defines a 3D volume
      [0,nx-1], [0,ny-1], [0,nz-1]
    - `level`: data value at which the isosurface is to be constructed
    - `nproc`: number of parallel processes to use. On multiprocessor machines
      this may be used to speed up the processing. If <= 0 , the number of
      processes will be set equal to the number of processors, to achieve
      a maximal speedup.

    Returns an (ntr,3,3) array defining the triangles of the isosurface.
    The result may be empty (if level is outside the data range).
    """
    if nproc < 1:
        nproc = cpu_count()

    if nproc == 1:
        # Perform single process isosurface (accelerated)
        from lib import misc
        data = data.astype(np.float32)
        level = np.float32(level)
        tri = misc.isosurface(data,level)

    else:
        # Perform parallel isosurface
        # 1. Split in blocks (and remember shift)
        datablocks = splitar(data,nproc,close=True)
        shift = (np.array([d.shape[0] for d in datablocks]) - 1).cumsum()
        # 2. Solve blocks independently
        tasks = [(isosurface,(d,level,1)) for d in datablocks]
        tri = multitask(tasks,nproc)
        # 3. Shift and merge blocks
        for t,s in zip(tri[1:],shift[:-1]):
            t[:,:,2] += s
        tri = np.concatenate(tri,axis=0)

    return tri



# End
