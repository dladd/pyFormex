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

"""Handling pyFormex Geometry Files

This module defines a class to work with files in the native
pyFormex Geometry File Format.
"""
from __future__ import print_function

import utils
import filewrite
from coords import *
from formex import Formex
from mesh import Mesh
from odict import ODict
from pyformex import message,debug,warning,DEBUG


import os

class GeometryFile(object):
    """A class to handle files in the pyFormex Geometry File format.

    The pyFormex Geometry File format allows the storage of most of the
    geometrical objects inside pyFormex, as well as some of their attributes.

    If `file` is a string, a file with that name is opened with the
    specified `mode`. If no mode is specified, 'r' will be used for
    existing files and 'w' for new files.
    Else, `file` should be an already open file.
    For files opened in write mode,

    Geometry classes can provide the facility 
    """

    _version_ = '1.5'

    def __init__(self,fil,mode=None,sep=' ',ifmt=' ',ffmt=' '):
        """Create the GeometryFile object."""
        isname = type(fil) == str
        if isname:
            if mode is None:
                if os.path.exists(fil):
                    mode = 'r'
                else:
                    mode = 'w'
            fil = open(fil,mode)
        self.isname = isname
        self.fil = fil
        self.writing = self.fil.mode[0] in 'wa'
        if self.writing:
            self.sep = sep
            self.fmt = {'i':ifmt,'f':ffmt}
        if self.isname:
            if mode[0] == 'w':
                self.writeHeader()
            elif mode[0] == 'r':
                self.readHeader()
            

    def reopen(self,mode='r'):
        """Reopen the file, possibly changing the mode.

        The default mode for the reopen is 'r'
        """
        self.fil.close()
        self.__init__(self.fil.name,mode)
        print(self.fil,self.writing,self.isname)


    def close(self):
        """Close the file.

        After closing, the file is no longer accessible.
        """
        self.fil.close()
        self.fil = None


    def checkWritable(self):
        if not self.writing:
            raise RuntimeError,"File is not opened for writing"
        

    def writeHeader(self):
        """Write the header of a pyFormex geometry file.

        The header identifies the file as a pyFormex geometry file
        and sets the following global values:

        - `version`: the version of the geometry file format
        - `sep`: the default separator to be used when not specified in
          the data block
        """
        self.fil.write("# pyFormex Geometry File (http://pyformex.org) version='%s'; sep='%s'\n" % (self._version_,self.sep))


    def writeData(self,data,sep,fmt=None):
        """Write an array of data to a pyFormex geometry file.

        If fmt is None, the data are written using numpy.tofile, with
        the specified separator.
        If fmt is specified,
        """
        if not self.writing:
            raise RuntimeError,"File is not opened for writing"
        kind = data.dtype.kind
        if fmt is None:
            fmt = self.fmt[kind]

        filewrite.writeData(data,self.fil,fmt)
            
        ## if fmt is None:
        ##     data.tofile(self.fil,sep)
        ##     self.fil.write('\n')
        ## else:
        ##     from lib.misc import tofile_int32,tofile_float32
        ##     val = data.reshape(-1,data.shape[-1])
        ##     if kind == 'i':
        ##         val = val.astype(int32)
        ##         tofile_int32(val,self.fil,'%i ')
        ##     elif kind == 'f':
        ##         val = val.astype(float32)
        ##         tofile_float32(val,self.fil,'%f ')
            

    def write(self,geom,name=None,sep=None):
        """Write any geometry object to the geometry file.

        `geom` is one of the Geometry data types of pyFormex or a list
        or dict of such objects.
        Currently exported geometry objects are
        :class:`Coords`, :class:`Formex`, :class:`Mesh`,
        :class:`PolyLine`, :class:`BezierSpline`.
        The geometry object is written to the file using the specified
        separator, or the default.
        """
        self.checkWritable()
        if isinstance(geom,dict):
            for name in geom:
                self.write(geom[name],name,sep)
        elif isinstance(geom,list):
            if name is None:
                for obj in geom:
                    self.write(obj,None,sep)
            else:
                name = utils.NameSequence(name)
                for obj in geom:
                    self.write(obj,name.next(),sep)
                    
        elif hasattr(geom,'write_geom'):
            geom.write_geom(self,name,sep)
        else:
            try:
                writefunc = getattr(self,'write'+geom.__class__.__name__)
            except:
                warning("Can not (yet) write objects of type %s to geometry file: skipping" % type(geom))
                return
            try:
                writefunc(geom,name,sep)
            except:
                warning("Error while writing objects of type %s to geometry file: skipping" % type(geom))


    def writeFormex(self,F,name=None,sep=None):
        """Write a Formex to the geometry file.

        `F` is a Formex. The coords attribute of the Formex is written as
        an array to the geometry file. If the Formex has a props attribute,
        it is also written.
        """
        if sep is None:
            sep = self.sep
        hasprop = F.prop is not None
        head = "# objtype='Formex'; nelems=%r; nplex=%r; props=%r; eltype=%r; sep=%r" % (F.nelems(),F.nplex(),hasprop,F.eltype,sep)
        if name:
            head += "; name='%s'" % name 
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        if hasprop:
            self.writeData(F.prop,sep)


    def writeMesh(self,F,name=None,sep=None,objtype='Mesh'):
        """Write a Mesh to a pyFormex geometry file.

        This writes a header line with these attributes and arguments:
        objtype, ncoords, nelems, nplex, props(True/False),
        eltype, name, sep.
        This is followed by the array data for: coords, elems, prop

        The objtype can/should be overridden for subclasses.
        """
        if objtype is None:
            objtype = 'Mesh'
        if sep is None:
            sep = self.sep
        hasprop = F.prop is not None
        head = "# objtype='%s'; ncoords=%s; nelems=%s; nplex=%s; props=%s; eltype='%s'; sep='%s'" % (objtype,F.ncoords(),F.nelems(),F.nplex(),hasprop,F.elName(),sep)
        if name:
            head += "; name='%s'" % name 
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        self.writeData(F.elems,sep)
        if hasprop:
            self.writeData(F.prop,sep)


    def writeTriSurface(self,F,name=None,sep=None):
        """Write a TriSurface to a pyFormex geometry file.

        This is equivalent to writeMesh(F,name,sep,objtype='TriSurface')
        """
        self.writeMesh(F,name=name,sep=sep,objtype='TriSurface')


    def writeCurve(self,F,name=None,sep=None,objtype=None,extra=None):
        """Write a Curve to a pyFormex geometry file.

        This function writes any curve type to the geometry file.
        The `objtype` is automatically detected but can be overridden.
        
        The following attributes and arguments are written in the header:
        ncoords, closed, name, sep.
        The following attributes are written as arrays: coords
        """
        if sep is None:
            sep = self.sep
        head = "# objtype='%s'; ncoords=%s; closed=%s; sep='%s'" % (F.__class__.__name__,F.coords.shape[0],F.closed,sep)
        if name:
            head += "; name='%s'" % name
        if extra:
            head += extra
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)


    def writePolyLine(self,F,name=None,sep=None):
        """Write a PolyLine to a pyFormex geometry file.

        This is equivalent to writeCurve(F,name,sep,objtype='PolyLine')
        """
        self.writeCurve(F,name=name,sep=sep,objtype='PolyLine')


    def writeBezierSpline(self,F,name=None,sep=None):
        """Write a BezierSpline to a pyFormex geometry file.

        This is equivalent to writeCurve(F,name,sep,objtype='BezierSpline')
        """
        self.writeCurve(F,name=name,sep=sep,objtype='BezierSpline',extra="; degree=%s" % F.degree)


    def writeNurbsCurve(self,F,name=None,sep=None,extra=None):
        """Write a NurbsCurve to a pyFormex geometry file.

        This function writes a NurbsCurve instance to the geometry file.
        
        The following attributes and arguments are written in the header:
        ncoords, nknots, closed, name, sep.
        The following attributes are written as arrays: coords, knots
        """
        if sep is None:
            sep = self.sep
        head = "# objtype='%s'; ncoords=%s; nknots=%s; closed=%s; sep='%s'" % (F.__class__.__name__,F.coords.shape[0],F.knots.shape[0],F.closed,sep)
        if name:
            head += "; name='%s'" % name
        if extra:
            head += extra
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        self.writeData(F.knots,sep)


    def writeNurbsSurface(self,F,name=None,sep=None,extra=None):
        """Write a NurbsSurface to a pyFormex geometry file.

        This function writes a NurbsSurface instance to the geometry file.
        
        The following attributes and arguments are written in the header:
        ncoords, nknotsu, nknotsv, closedu, closedv, name, sep.
        The following attributes are written as arrays: coords, knotsu, knotsv
        """
        if sep is None:
            sep = self.sep
        head = "# objtype='%s'; ncoords=%s; nuknots=%s; nvknots=%s; uclosed=%s; vclosed=%s; sep='%s'" % (F.__class__.__name__,F.coords.shape[0],F.uknots.shape[0],F.uknots.shape[0],F.closed[0],F.closed[1],sep)
        if name:
            head += "; name='%s'" % name
        if extra:
            head += extra
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        self.writeData(F.uknots,sep)
        self.writeData(F.vknots,sep)


    def readHeader(self):
        """Read the header of a pyFormex geometry file.

        """
        sep = ' '
        s = self.fil.readline()
        if s.startswith('# Formex'):
            version = '1.1'
        elif s.startswith('# pyFormex Geometry File'):
            pos = s.rfind(')')
            exec(s[pos+1:].strip())
        else:
            version = None
            raise RuntimeError,"This does not look like a pyFormex geometry file, or it is a very old version."
            
        self._version_ = version
        self.sep = sep
        self.objname = utils.NameSequence('%s_000' % utils.projectName(self.fil.name))
        self.results = ODict()


    def read(self,count=-1):
        """Read a pyFormex Geometry File.

        fil is a filename or a file object.
        If the file is in a valid pyFormex geometry file format, geometry
        objects are read from the file and returned in a dictionary.
        The object names are used as keys. If the file does not contain
        object names, they will be autogenerated from the file name.

        A count may be specified to limit the number of objects read.

        If the file format is invalid and no valid geometry could be read,
        None is returned.
        Valid pyFormex geometry file formats are described in the manual.
        """
        eltype = None # for compatibility with pre 1.1 .formex files
        ndim = 3
        while True:
            objtype = 'Formex' # the default obj type
            obj = None
            sep = self.sep
            name = None
            s = self.fil.readline()
            
            if len(s) == 0:   # end of file
                break

            if not s.startswith('#'):  # not a header: skip
                continue


            ###
            ###  THIS WILL USE UNDEFINED VALUES FROM A PREVIOUS OBJECT
            ###  THIS SHOULD THEREFORE BE CHANGED !!!!
            ###
            try:
                exec(s[1:].strip())
            except:
                continue  # not a legal header: skip

            debug("Reading object of type %s" % objtype,DEBUG.INFO) 

            # OK, we have a legal header, try to read data
            if objtype == 'Formex':
                obj = self.readFormex(nelems,nplex,props,eltype,sep)
            elif objtype in ['Mesh','TriSurface']:
                obj = self.readMesh(ncoords,nelems,nplex,props,eltype,sep,objtype)
            elif objtype == 'PolyLine':
                obj = self.readPolyLine(ncoords,closed,sep)
            elif objtype == 'BezierSpline':
                if 'nparts' in s:
                    # THis looks like a version 1.3 BezierSpline
                    obj = self.oldReadBezierSpline(ncoords,nparts,closed,sep)
                else:
                    if not 'degree' in s:
                        # compatibility with 1.4  BezierSpline records
                        degree = 3
                    obj = self.readBezierSpline(ncoords,closed,degree,sep)
            elif objtype == 'NurbsCurve':
                obj = self.readNurbsCurve(ncoords,nknots,closed,sep)
            elif objtype in globals() and hasattr(globals()[objtype],'read_geom'):
                obj = globals()[objtype].read_geom(self)
            else:
                message("Can not (yet) read objects of type %s from geometry file: skipping" % objtype)
                continue # skip to next header


            if obj is not None:
                if name is None:
                    name = self.objname.next()
                self.results[name] = obj

            if count > 0 and len(self.results) >= count:
                break

        if self.isname:
            self.fil.close()

        return self.results
        

    def readFormex(self,nelems,nplex,props,eltype,sep):
        """Read a Formex from a pyFormex geometry file.

        The coordinate array for nelems*nplex points is read from the file.
        If present, the property numbers for nelems elements are read.
        From the coords and props a Formex is created and returned.
        """
        ndim = 3
        f = readArray(self.fil,Float,(nelems,nplex,ndim),sep=sep)
        if props:
            p = readArray(self.fil,Int,(nelems,),sep=sep)
        else:
            p = None
        return Formex(f,p,eltype)
 

    def readMesh(self,ncoords,nelems,nplex,props,eltype,sep,objtype='Mesh'):
        """Read a Mesh from a pyFormex geometry file.

        The following arrays are read from the file:
        - a coordinate array with `ncoords` points,
        - a connectivity array with `nelems` elements of plexitude `nplex`,
        - if present, a property number array for `nelems` elements.

        Returns the Mesh constructed from these data, or a subclass if
        an objtype is specified.
        """
        # Make sure to import the Mesh subclasses that can be read
        from plugins.trisurface import TriSurface
        
        ndim = 3
        x = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        e = readArray(self.fil,Float,(nelems,nplex),sep=sep)
        if props:
            p = readArray(self.fil,Int,(nelems,),sep=sep)
        else:
            p = None
        M = Mesh(x,e,p,eltype)
        if objtype != 'Mesh':
            try:
                clas = locals()[objtype]
            except:
                clas = globals()[objtype]
            M = clas(M)
        return M
 

    def readPolyLine(self,ncoords,closed,sep):
        """Read a Curve from a pyFormex geometry file.

        The coordinate array for ncoords points is read from the file
        and a Curve of type `objtype` is returned.
        """
        from plugins.curve import PolyLine
        ndim = 3
        coords = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        return PolyLine(control=coords,closed=closed)
 

    def readBezierSpline(self,ncoords,closed,degree,sep):
        """Read a BezierSpline from a pyFormex geometry file.

        The coordinate array for ncoords points is read from the file
        and a BezierSpline of the given degree is returned.
        """
        from plugins.curve import BezierSpline
        ndim = 3
        coords = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        return BezierSpline(control=coords,closed=closed,degree=degree)
 

    def readNurbsCurve(self,ncoords,nknots,closed,sep):
        """Read a NurbsCurve from a pyFormex geometry file.

        The coordinate array for ncoords control points and the nknots
        knot values are read from the file.
        A NurbsCurve of degree p = nknots - ncoords - 1 is returned.
        """
        from plugins.nurbs import NurbsCurve
        ndim = 4
        coords = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        knots = readArray(self.fil,Float,(nknots,),sep=sep)
        return NurbsCurve(control=coords,knots=knots,closed=closed)
 

    def readNurbsSurface(self,ncoords,nuknots,nvknots,uclosed,vclosed,sep):
        """Read a NurbsSurface from a pyFormex geometry file.

        The coordinate array for ncoords control points and the nuknots and
        nvknots values of uknots and vknots are read from the file.
        A NurbsSurface of degree ``pu = nuknots - ncoords - 1``  and
        ``pv = nvknots - ncoords - 1`` is returned.
        """
        from plugins.nurbs import NurbsSurface
        ndim = 4
        coords = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        uknots = readArray(self.fil,Float,(nuknots,),sep=sep)
        vknots = readArray(self.fil,Float,(nvknots,),sep=sep)
        return NurbsSurface(control=coords,knots=(uknots,vknots),closed=(uclosed,vclosed))


    def oldReadBezierSpline(self,ncoords,nparts,closed,sep):
        """Read a BezierSpline from a pyFormex geometry file version 1.3.

        The coordinate array for ncoords points and control point array
        for (nparts,2) control points are read from the file.
        A BezierSpline is constructed and returned.
        """
        from plugins.curve import BezierSpline
        ndim = 3
        coords = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        control = readArray(self.fil,Float,(nparts,2,ndim),sep=sep)
        return BezierSpline(coords,control=control,closed=closed)


    def rewrite(self):
        """Convert the geometry file to the latest format.
       
        The conversion is done by reading all objects from the geometry file
        and writing them back. Parts that could not be succesfully read will
        be skipped.
        """
        self.reopen('r')
        obj = self.read()
        self._version_ = GeometryFile._version_
        #print self._version_
        if obj is not None:
            self.reopen('w')
            self.write(obj)
        self.close()

# End
