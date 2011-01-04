# $Id$

"""Write geometry to file in a whole number of formats.

This module defines bothe the basic routines to write geometrical data
to a file and the specialized exporters to write files in a number of
well known standardized formats.

The basic routines are very versatile as well as optimized (using the version
in the pyFormex C-library) and allow to easily create new exporters for
other formats.


"""
import pyformex as pf
from numpy import int32,float32
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
            misc.tofile_int32(val.astype(int32),fil,'%i ')
        elif kind == 'f':
            misc.tofile_float32(val.astype(float32),fil,'%f ')
        else:
            raise ValueError,"Can not write data fo type %s" % data.dtype


# Output of surface file formats

def writeGTS(fn,coords,edges,faces):
    if coords.shape[1] != 3 or edges.shape[1] != 2 or faces.shape[1] != 3:
        raise runtimeError, "Invalid arguments or shape"
    fil = file(fn,'w')
    fil.write("%d %d %d\n" % (coords.shape[0],edges.shape[0],faces.shape[0]))
    writeData(coords,fil,'%f ')
    writeData(edges+1,fil,'%i ')
    writeData(faces+1,fil,'%i ')
    fil.write("#GTS file written by %s\n" % pf.Version)
    fil.close()


# End



