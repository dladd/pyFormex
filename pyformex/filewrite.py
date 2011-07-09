# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""Write geometry to file in a whole number of formats.

This module defines bothe the basic routines to write geometrical data
to a file and the specialized exporters to write files in a number of
well known standardized formats.

The basic routines are very versatile as well as optimized (using the version
in the pyFormex C-library) and allow to easily create new exporters for
other formats.


"""
import pyformex as pf
import numpy as np
from lib import misc


def writeData(data,fil,fmt=' '):
    """Write an array of numerical data to an open file.

    Parameters:

    - `data`: a numerical array of int or float type
    - `fil`: an open file object
    - `fmt`: a format string defining how a single data item is written.
      It should be one of:

      - '': an empty string: in this case the data are written in binary
        mode, using the function numpy.tofile.
      - ' ': a single space: in this case the data are written in text
        mode, separated by a space, also using the function numpy.tofile.
        At the end, a newline is added. All the data of the array thus appear
        on a single line. 
      - a format string compatible with the array data type. In this case
        float arrays will be forced to float32 and int arrays to int32.
        The format string should contain a valid format converter for a
        a single data item in both Python and C. They should also contain
        the necessary spacing or separator. Examples are '%5i ' for int data
        and '%f,' or '%10.3e' for float data. The array will be converted
        to a 2D array, keeping the lengt of the last axis. The all elements
        will be written row by row using the specified format string, and a
        newline character will be written after each row.
        This mode is written by pyFormex function misc.tofile_int32 or
        misc.tofile_float32, which have accelerated versions in the pyFormex
        C library.
    """
    kind = data.dtype.kind
    if fmt == '' or fmt == ' ':
        data.tofile(fil,sep=fmt)
        if fmt == ' ':
            fil.write('\n')
    else:
        val = data.reshape(-1,data.shape[-1])
        if kind == 'i':
            misc.tofile_int32(val.astype(np.int32),fil,fmt)
        elif kind == 'f':
            misc.tofile_float32(val.astype(np.float32),fil,fmt)
        else:
            raise ValueError,"Can not write data fo type %s" % data.dtype


def writeIData(data,fil,fmt,ind=1):
    """Write an indexed array of numerical data to an open file.

    ind = i: autoindex from i
          array: use these indices  
    """
    kind = data.dtype.kind
    val = data.reshape(-1,data.shape[-1])
    nrows = val.shape[0]
    if type(ind) is int:
        ind = ind + np.arange(nrows)
    else:
        ind = ind.reshape(-1)
        if ind.shape[0] != nrows:
            raise ValueError,"Index should have same length as data"
        
    if kind == 'i':
        raise ImplementationError
        misc.tofile_int32(val.astype(np.int32),fil,fmt)
    elif kind == 'f':
        misc.tofile_ifloat32(ind.astype(np.int32),val.astype(np.float32),fil,fmt)
    else:
        raise ValueError,"Can not write data fo type %s" % data.dtype


# Output of mesh file formats

def writeOFF(fn,coords,elems):
    """Write a mesh of polygons to a file in OFF format.

    Parameters:

    - `fn`: file name, by preference ending on '.off'
    - `coords`: float array with shape (ncoords,3), with the coordinates of
      `ncoords` vertices.
    - `elems`: int array with shape (nelems,nplex), with the definition of
      `nelems` polygon elements.
    """
    if coords.dtype.kind != 'f' or coords.ndim != 2 or coords.shape[1] != 3 or elems.dtype.kind != 'i' or elems.ndim != 2:
        raise runtimeError, "Invalid type or shape of argument(s)"
    
    fil = file(fn,'w')
    fil.write("OFF\n")
    fil.write("%d %d 0\n" % (coords.shape[0],elems.shape[0]))
    writeData(coords,fil,'%f ')
    nelems = np.zeros_like(elems[:,:1])
    nelems.fill(elems.shape[1])
    elemdata = np.column_stack([nelems,elems])
    writeData(elemdata,fil,'%i ')
    fil.close()


# Output of surface file formats

def writeGTS(fn,coords,edges,faces):
    """Write a mesh of triangles to a file in GTS format.

    Parameters:

    - `fn`: file name, by preference ending on '.gts'
    - `coords`: float array with shape (ncoords,3), with the coordinates of
      `ncoords` vertices.
    - `edges`: int array with shape (nedges,2), with the definition of
      `nedges` edges in function of the vertex indices.
    - `faces`: int array with shape (nfaces,3), with the definition of
      `nfaces` triangles in function of the edge indices.
    """
    if coords.dtype.kind != 'f' or coords.ndim != 2 or coords.shape[1] != 3 or edges.dtype.kind != 'i' or edges.ndim != 2 or edges.shape[1] != 2 or faces.dtype.kind != 'i' or faces.ndim != 2 or faces.shape[1] != 3:
        raise runtimeError, "Invalid type or shape of argument(s)"

    fil = file(fn,'w')
    fil.write("%d %d %d\n" % (coords.shape[0],edges.shape[0],faces.shape[0]))
    writeData(coords,fil,'%f ')
    writeData(edges+1,fil,'%i ')
    writeData(faces+1,fil,'%i ')
    fil.write("#GTS file written by %s\n" % pf.Version)
    fil.close()


# End



