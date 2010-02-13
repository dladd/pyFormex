# $Id$

"""Handling pyFormex Geometry Files

This module defines a class to work with files in the native
pyFormex Geometry File Format.
"""

import utils
from coords import *
from formex import Formex
from plugins.mesh import Mesh
from plugins.curve import Curve
from odict import ODict

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
    """

    _version_ = '1.2'

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
        self.writing = fil.mode[0] in 'wa'
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
        self.fil = open(self.fil.name,mode)


    def close(self):
        """Close the file.

        After closing, the file is no longer accessible.
        """
        self.fil.close()
        self.fil = None


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
        data.tofile(self.fil,sep)
        self.fil.write('\n')
        

    def write(self,geom,name=None,sep=None):
        """Write any geometry object to the geometry file.

        `geom` is one of the Geometry data types of pyFormex or a list
        or dict of such objects.
        Currently exported geometry objects are
        :class:`Coords`, :class:`Formex`, :class:`Mesh`, :class:`Curve`.
        The geometry object is written to the file using the specified
        separator, or the default.
        """
        if isinstance(geom,dict):
            for name in geom:
                self.write(geom[name],name,sep)
        elif isinstance(geom,list):
            for obj in geom:
                self.write(obj,None,sep)
        elif isinstance(geom,Formex):
            self.writeFormex(geom,name,sep)
        elif isinstance(geom,Mesh):
            self.writeMesh(geom,name,sep)


    def writeFormex(self,F,name=None,sep=None):
        """Write a Formex to the geometry file.

        `F` is a Formex. The coords attribute of the Formex is written as
        an array to the geometry file. If the Formex has a props attribute,
        it is also written.
        """
        if sep is None:
            sep = self.sep
        hasprop = F.prop is not None
        head = "# nelems=%r; nplex=%r; props=%r; eltype=%r; sep='%s'" % (F.nelems(),F.nplex(),hasprop,F.eltype,sep)
        if name:
            head += "; name='%s'" % name 
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        if hasprop:
            self.writeData(F.prop,sep)


    def writeMesh(self,F,name=None,sep=None):
        """Write a Mesh to the geometry file.

        `F` is a Mesh. The following attributes of the Mesh are written as
        arrays to the geometry file: coords, elems, prop
        """
        if sep is None:
            sep = self.sep
        hasprop = F.prop is not None
        head = "# objtype='Mesh'; ncoords=%r; nelems=%r; nplex=%r; props=%r; eltype=%r; sep='%s'" % (F.ncoords(),F.nelems(),F.nplex(),hasprop,F.eltype,sep)
        if name:
            head += "; name='%s'" % name 
        self.fil.write(head+'\n')
        self.writeData(F.coords,sep)
        self.writeData(F.elems,sep)
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
        If the file is in a valid Formex file format, the Formex is read and
        returned. Otherwise, None is returned.
        Valid Formex file formats are described in the manual.
        """
        eltype = None # for compatibility with pre 1.1 .formex files
        ndim = 3
        while True:
            objtype = 'Formex' # the default obj type
            obj = None
            sep = self.sep
            name = None
            s = self.fil.readline()
            if len(s) == 0:
                break

            if s.startswith('#'):
                try:
                    exec(s[1:].strip())
                except:
                    continue

            if objtype == 'Formex':
                obj = self.readFormex(nelems,nplex,props,eltype,sep)
            elif objtype == 'Mesh':
                obj = self.readMesh(ncoords,nelems,nplex,props,eltype,sep)


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
        
    
# End
