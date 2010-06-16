# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

"""Handling pyFormex Geometry Files

This module defines a class to work with files in the native
pyFormex Geometry File Format.
"""

import utils
from coords import *
from formex import Formex
from plugins.mesh import Mesh
from plugins.curve import PolyLine,BezierSpline
from odict import ODict
from pyformex import message

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

    _version_ = '1.3'

    def __init__(self,fil,mode=None,sep=' '):
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
        print self.fil,self.writing,self.isname


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


    def writeData(self,data,sep):
        """Write an array of data to a pyFormex geometry file."""
        if not self.writing:
            raise RuntimeError,"File is not opened for writing"
        data.tofile(self.fil,sep)
        self.fil.write('\n')
        

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
                    
        elif isinstance(geom,Formex):
            self.writeFormex(geom,name,sep)
        elif hasattr(geom,'write_geom'):
            geom.write_geom(self,name,sep)
        else:
            message("Can not (yet) write objects of type %s to geometry file: skipping" % type(geom))


    def writeFormex(self,F,name=None,sep=None):
        """Write a Formex to the geometry file.

        `F` is a Formex. The coords attribute of the Formex is written as
        an array to the geometry file. If the Formex has a props attribute,
        it is also written.
        """
        if sep is None:
            sep = self.sep
        hasprop = F.prop is not None
        head = "# objtype='Formex'; nelems=%r; nplex=%r; props=%r; eltype=%r; sep='%s'" % (F.nelems(),F.nplex(),hasprop,F.eltype,sep)
        if name:
            head += "; name='%s'" % name 
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        if hasprop:
            self.writeData(F.prop,sep)


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
            
            try:
                exec(s[1:].strip())
            except:
                continue  # not a legal header: skip

            print("READING OBJECT OF TYPE %s" % objtype) 

            # OK, we have a legal header, try to read data
            if objtype == 'Formex':
                obj = self.readFormex(nelems,nplex,props,eltype,sep)
            elif objtype == 'Mesh':
                obj = self.readMesh(ncoords,nelems,nplex,props,eltype,sep)
            elif objtype == 'PolyLine':
                obj = PolyLine.read_geom(self,ncoords,closed,sep)
            elif objtype == 'BezierSpline':
                obj = BezierSpline.read_geom(self,ncoords,nparts,closed,sep)
            elif globals().has_key(objtype) and hasattr(globals()[objtype],'read_geom'):
                obj = globals()[objtype].read_geom(self)
            else:
                from pyformex import message
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
        ndim = 3
        f = readArray(self.fil,Float,(nelems,nplex,ndim),sep=sep)
        if props:
            p = readArray(self.fil,Int,(nelems,),sep=sep)
        else:
            p = None
        return Formex(f,p,eltype)
 

    def readMesh(self,ncoords,nelems,nplex,props,eltype,sep):
        ndim = 3
        x = readArray(self.fil,Float,(ncoords,ndim),sep=sep)
        e = readArray(self.fil,Float,(nelems,nplex),sep=sep)
        if props:
            p = readArray(self.fil,Int,(nelems,),sep=sep)
        else:
            p = None
        return Mesh(x,e,p,eltype)


    def rewrite(self):
        """Convert the geometry file to the latest format.
       
        The conversion is done by reading all objects from the geometry file
        and writing them back. Parts that could not be succesfully read will
        be skipped.
        """
        self.reopen('r')
        obj = self.read()
        self._version_ = GeometryFile._version_
        print self._version_
        if obj is not None:
            self.reopen('w')
            self.write(obj)
        self.close()

# End
