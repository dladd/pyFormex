#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
#
"""Numerical data reader"""

__all__ = ['splitFloat','readData']

import re
from numpy import *

Float = re.compile('[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE]\d+)?')
FloatString = re.compile('(?P<float>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE]\d+)?)(?P<string>.*)')
        

def splitFloat(s):
    """Match a floating point number at the beginning of a string

    If the beginning of the string matches a floating point number,
    a list is returned with the float and the remainder of the string;
    if not, None is returned.
    Example: ``splitFloat('123e4rt345e6')`` returns ``[1230000.0, 'rt345e6']``
    """
    m = FloatString.match(s)
    if m:
        return [float(m.group('float')), m.group('string')]
    return None


def readData(s,type,strict=False):
    """Read data from a line matching the 'type' specification.

    This is a powerful function for reading, interpreting and converting
    numerical data from a string. Fields in the string s are separated by
    commas. The 'type' argument is a list where each element specifies how
    the corresponding field should be interpreted. Available values are
    'int', 'float' or some unit ('kg', 'm', etc.).
    If the type field is 'int' or 'float', the data field is converted to
    the matching type.
    If the type field is a unit, the data field should be a number and a
    unit separated by space or not, or just a number. If it is just a number,
    its value is returned unchanged (as float). If the data contains a unit,
    the number is converted to the requested unit. It is an error if the
    datafield holds a non-conformable unit.
    The function returns a list of ints and/or floats (without the units).
    If the number of data fields is not equal to the number of type specifiers,
    the returned list will correspond to the shortest of both and the surplus
    data or types are ignored, UNLESS the strict flag has been set, in which
    case a RuntimError is raised.
    Example::
    
       readData('12, 13, 14.5e3, 12 inch, 1hr, 31kg ', ['int','float','kg','cm','s'])
      
    returns ``[12, 13.0, 14500.0, 30.48, 3600.0]``
    
    ..warning ::
    
      You need to have the GNU ``units`` command installed for the unit
      conversion to work. 
    """
    import units,string
    out = []
    data = string.split(s,',')
    if strict and len(data) != len(type):
        raise RuntimeError, "Data do not match type specifier %s\nData: '%s'" % (type,s)
    for t,d in zip(type,data):
        #print(t,d)
        if len(d) == 0:
            break
        v = string.strip(d)
        if t == 'int':
            val = int(v)
        elif t == 'float':
            val = float(v)
        else:
            m = Float.match(v)
            #print(m.start(),m.end(),len(v))
            if m and m.end() == len(v):
                val = float(v)
            else:
                val = float(units.ConvertUnits(v,t))
        out.append(val)
    return out


###################################################
## BV: The following functions need clean up
## or should be replaced with a common module with calpy
##


def readAsciiTable(fn, header=True):
    """_Reads data from an ASCII text file (Table).
    
    if header is True: first line is the header, the rest is a table (2D array)
    it returns the header and the 2D array of data as floats.
    if header is False, there is no header.
    it returns the header as None and the 2D array of data as floats.
    """
    fil=open(fn,'r')
    line = fil.readline()
    h = line.strip('\n').split()
    data = fromfile(fil,sep=' ',dtype=float32)
    if header==False:
        h=asarray(h, dtype=float)
        data=append(h, data)
        return None, data.reshape (-1,len(h) )
    if header==True:
        data = data.reshape((-1,len(h)))
    return h, data

def writeAsciiTable(fn, h, d, fmtdata='e'):
    """_Writes an ASCII text file.
    
    The first line is the header h (tuple of strings, e.g. h=[ 'n0', 'n1', 'n2' ] ).
    The other lines contains the data d (2D array of floats) as a table. 
    If format of the data can be chosen with fmtdata:
        1) fmtdata is 'e' , the data are written as scientific (1.123456e+01),
        2) fmtdata is 'f' , the data are written as floats,
        3)fmtdata is 'd' , the data are written as integers,
        4) otherwise, the fmt is specified (e.g. fmtdata='d f f d'). In this case, there should be as many fmt as data columns"""
    fil=open(fn,'w')
    ncol= len(h)-1#number of columns-1
    fmtH = '%s'+ncol*' %s' + '\n'
    fil.write(fmtH %(tuple(h )))#header
    if fmtdata==None:fmtdata='e'
    if fmtdata=='e':
        fmtD = '%e'+ncol*' %e' + '\n'
    elif fmtdata=='f':
        fmtD = '%f'+ncol*' %f' + '\n'
    elif fmtdata=='d':
        fmtD = '%d'+ncol*' %d' + '\n'
    else:
        dt= fmtdata.split(' ')
        sdt='%'+dt[0]
        for t in dt[1:]:
            sdt=sdt+' %'+t
        fmtD=sdt+'\n'
    for ld in d:
        fil.write(fmtD % (tuple(ld)))#line of data
    fil.close()




if __name__ == "__main__":
    print(readData('12, 13, 14.5e3, 12 inch, 1hr, 5MPa', ['int','float','kg','cm','s']))
