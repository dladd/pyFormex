# $Id$    pyformex 
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

"""project.py

Functions for managing a project in pyFormex.
"""
from __future__ import print_function
import pyformex as pf
from track import TrackedDict
from pyformex import utils
import os,sys
import cPickle
import gzip

_signature_ = pf.FullVersion

module_relocations = {
    'plugins.mesh' : 'mesh',
    'plugins.surface' : 'plugins.trisurface',
}

def find_global(module,name):
    """Override the import path of some classes"""
    pf.debug("I want to import %s from %s" % (name,module),pf.DEBUG.PROJECT)
    if module in module_relocations:
        module = module_relocations[module]
        pf.debug("  I will try module %s instead" % module,pf.DEBUG.PROJECT)
    __import__(module)
    mod = sys.modules[module]
    clas = getattr(mod, name)
    return clas        
    

def pickle_load(f,try_resolve=True):
    """Load data from pickle file f."""
    pi = cPickle.Unpickler(f)
    if try_resolve:
        pi.find_global = find_global

    return pi.load()

highest_format = 3

class Project(TrackedDict):
    """Project: a persistent storage of pyFormex data.

    A pyFormex Project is a regular Python dict that can contain named data
    of any kind, and can be saved to a file to create persistence over
    different pyFormex sessions.

    The :class:`Project` class is used by pyFormex for the ``pyformex.PF``
    global variable that collects variables exported from pyFormex scripts.
    While projects are mostly handled through the pyFormex GUI, notably the
    *File* menu, the user may also create and handle his own Project objects
    from a script.

    Because of the way pyFormex Projects are written to file,
    there may be problems when trying to read a project file that was
    created with another pyFormex version. Problems may occur if the
    project contains data of a class whose implementation has changed,
    or whose definition has been relocated. Our policy is to provide
    backwards compatibility: newer versions of pyFormex will normally
    read the older project formats. Saving is always done in the
    newest format, and these can generally not be read back by older
    program versions (unless you are prepared to do some hacking).

    .. warning:: Compatibility issues.

       Occasionally you may run into problems when reading back an
       old project file, especially when it was created by an unreleased
       (development) version of pyFormex. Because pyFormex is evolving fast,
       we can not test the full compatibility with every revision
       You can file a support request on the pyFormex `support tracker`_.
       and we will try to add the required conversion code to
       pyFormex.

       The project files are mainly intended as a means to easily save lots
       of data of any kind and to restore them in the same session or a later
       session, to pass them to another user (with the same or later pyFormex
       version), to store them over a medium time period. Occasionally opening
       and saving back your project files with newer pyFormex versions may help
       to avoid read-back problems over longer time.

       For a problemless long time storage of Geometry type objects you may
       consider to write them to a pyFormex Geometry file (.pgf) instead, since
       this uses a stable ascii based format. It can (currently) not deal
       with other data types however.

    Parameters:

    - `filename`: the name of the file where the Project data will be saved.
      If the file exists (and `access` is not `w`), it should be a previously
      saved Project and an attempt will be made to load the data from this
      file into the Project.
      If this fails, an error is raised. 

      If the file exists and `access` is `w`, it will be overwritten,
      destroying any previous contents.

      If no filename is specified, a temporary file will be created when
      the Project is saved for the first time. The file with not be
      automatically deleted. The generated name can be retrieved from the
      filename attribute.

    - `access`: One of 'wr' (default), 'rw', 'w' or 'r'.
      If the string contains an 'r' the data from an existing file will be
      read into the dict. If the string starts with an 'r', the file should
      exist. If the string contains a 'w', the data can be written back to
      the file. The 'r' access mode is thus a read-only mode.
      
      ======  ===============  ============  ===================
      access  File must exist  File is read  File can be written
      ======  ===============  ============  ===================
        r           yes             yes             no          
        rw          yes             yes             yes         
        wr          no         if it exists         yes         
        w           no              no              yes         
      ======  ===============  ============  ===================

    - `convert`: if True (default), and the file is opened for reading, an
      attempt is made to open old projects in a compatibility mode, doing the
      necessary conversions to new data formats. If convert is set False,
      only the latest format can be read and older formats will generate
      an error.
    
    - `signature`: A text that will be written in the header record of the
      file. This can e.g. be used to record format version info.

    - `compression`: An integer from 0 to 9: compression level. For large
      data sets, compression leads to much smaller files. 0 is no compression,
      9 is maximal compression. The default is 4.

    - `binary`: if False and no compression is used, storage is done
      in an ASCII format, allowing to edit the file. Otherwise, storage
      uses a binary format. Using binary=False is deprecated.

    - `data`: a dict-like object to initialize the Project contents. These data
      may override values read from the file.  

    Example:
    
      >>> d = dict(a=1,b=2,c=3,d=[1,2,3],e={'f':4,'g':5})
      >>> import tempfile
      >>> f = tempfile.mktemp('.pyf','w')
      >>> P = Project(f)
      >>> P.update(d)
      >>> print dict.__str__(P)
      {'a': 1, 'c': 3, 'b': 2, 'e': {'g': 5, 'f': 4}, 'd': [1, 2, 3]}
      >>> P.save(quiet=True)
      >>> P.clear()
      >>> print dict.__str__(P)
      {}
      >>> P.load(quiet=True)
      >>> print dict.__str__(P)
      {'a': 1, 'c': 3, 'b': 2, 'e': {'g': 5, 'f': 4}, 'd': [1, 2, 3]}

    """

    def __init__(self,filename=None,access='wr',convert=True,signature=_signature_,compression=5,binary=True,data={},**kargs):
        """Create a new project."""
        if 'create' in kargs:
            utils.warn("The create=True argument should be replaced with access='w'")
        if 'legacy' in kargs:
            utils.warn("The legacy=True argument has become superfluous")
            
        self.filename = filename
        self.access = access
        self.signature = str(signature)
        self.gzip = compression if compression in range(1,10) else 0
        self.mode = 'b' if binary or compression > 0 else ''
        
        TrackedDict.__init__(self)
        if filename and os.path.exists(filename) and 'r' in self.access:
            # read existing contents
            self.load(convert)
            self.hits = 0
        if data:
            self.update(data)
        if filename and access=='w':
            # destroy existing contents
            self.save()
        pf.debug("INITIAL hits = %s" % self.hits,pf.DEBUG.PROJECT)


    def __str__(self):
        s = """Project name: %s
  access: %s    mode: %s     gzip:%s
  signature: %s
  contents: %s
""" % (self.filename,self.access,self.mode,self.gzip,self.signature,
        self.contents())
        return s


    def contents(self):
        k = self.keys()
        k.sort()
        return k


    def header_data(self):
        """Construct the data to be saved in the header."""
        store_attr = ['signature','gzip','mode','autofile','_autoscript_']
        store_vals = [getattr(self,k,None) for k in store_attr]
        return dict([(k,v) for k,v in zip(store_attr,store_vals) if v is not None])


    def save(self,quiet=False):
        """Save the project to file."""
        if 'w' not in self.access:
            pf.debug("Not saving because Project file opened readonly",pf.DEBUG.PROJECT)
            return

        if not quiet:
            print("Project variables changed: %s" % self.hits)

        if self.filename is None:
            import tempfile
            fd,fn = tempfile.mkstemp(prefix='pyformex_',suffix='.pyf')
            self.filename = fn
        else:
            if not quiet:
                print("Saving project %s with mode %s and compression %s" % (self.filename,self.mode,self.gzip))
            #print("  Contents: %s" % self.keys()) 
        f = open(self.filename,'w'+self.mode)
        # write header
        # self.signature = pf.FullVersion
        f.write("%s\n" % self.header_data())
        f.flush()
        if self.mode == 'b':
            # When using binary, can as well use highest protocol
            protocol = cPickle.HIGHEST_PROTOCOL
        else:
            protocol = 0
        if self.gzip:
            pyf = gzip.GzipFile(mode='w'+self.mode,compresslevel=self.gzip,fileobj=f)
            cPickle.dump(self,pyf,protocol)
            pyf.close()
        else:
            cPickle.dump(self,f,protocol)
        f.close()
        self.hits = 0


    def readHeader(self,quiet=False):
        """Read the header from a project file.
        
        Tries to read the header from different legacy formats,
        and if succesfull, adjusts the project attributes according
        to the values in the header.
        Returns the open file if succesfull.
        """
        self.format = -1
        if not quiet:
            print("Reading project file: %s" % self.filename)
        f = open(self.filename,'rb')
        fpos = f.tell()
        s = f.readline()
        # Try subsequent formats
        try:
            # newest format has header in text format
            header = eval(s)
            self.__dict__.update(header)
            self.format = 3
        except:

            # try OLD new format: the first pickle contains attributes
            try:
                p = pickle_load(f)
                self.__dict__.update(p)
                self.format = 2
            except:
                s = s.strip()
                if not quiet:
                    print("Header = '%s'" % s)
                if s=='gzip' or s=='' or 'pyFormex' in s:
                    # transitional format
                    self.gzip = 5
                    self.format = 1
                    # NOT SURE IF THIS IS OK, NEED EXAMPLE FILE
                    f.seek(fpos)
                else:
                    # headerless format
                    f.seek(0)
                    self.gzip = 0
                    self.format = 0

        return f


    def load(self,try_resolve=False,quiet=False):
        """Load a project from file.
        
        The loaded definitions will update the current project.
        """
        f = self.readHeader(quiet)
        if self.format < highest_format:
            if not quiet:
                print("Format looks like %s" % self.format)
            utils.warn('warn_old_project')
        with f:
            try:
                if not quiet:
                    print("Unpickling gzip")
                pyf = gzip.GzipFile(fileobj=f,mode='rb')
                p = pickle_load(pyf,try_resolve)
                pyf.close()
            except:
                if not quiet:
                    print("Unpickling clear")
                p = pickle_load(f,try_resolve)
            self.update(p)
    

    def convert(self,filename=None):
        """Convert an old format project file.
        
        The project file is read, and if successfull, is immediately
        saved. By default, this will overwrite the original file.
        If a filename is specified, the converted data are saved to
        that file.
        In both cases, access is set to 'wr', so the tha saved data can
        be read back immediately.
        """
        self.load(True)
        print("GOT KEYS %s" % self.keys())
        if filename is not None:
            self.filename = filename
        self.access = 'w'
        print("Will now save to %s" % self.filename)
        self.save()
    

    def uncompress(self):
        """Uncompress a compressed project file.
        
        The project file is read, and if successfull, is written
        back in uncompressed format. This allows to make conversions
        of the data inside.
        """
        f = self.readHeader()
        print(self.format,self.gzip)
        if f:
            if self.gzip:
                try: 
                    pyf = gzip.GzipFile(self.filename,'r',self.gzip,f)
                except:
                    self.gzip = 0

            if self.gzip:
                fn = self.filename.replace('.pyf','_uncompressed.pyf')
                fu = open(fn,'w'+self.mode)
                h = self.header_data()
                h['gzip'] = 0
                fu.write("%s\n" % h)
                while True:
                    x = pyf.read()
                    if x:
                        fu.write(x)
                    else:
                        break
                fu.close()
                print("Uncompressed %s to %s" % (self.filename,fn))
                
            else:    
                utils.warn("The contents of the file does not appear to be compressed.")
            f.close()

 
    def delete(self):
        """Unrecoverably delete the project file."""
        os.remove(self.filename)


# End
