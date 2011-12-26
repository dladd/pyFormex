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

"""project.py

Functions for managing a project in pyFormex.
"""

import os,sys
import cPickle
import gzip

_signature_ = 'pyFormex 0.8.6'

module_relocations = {
    'plugins.mesh' : 'mesh',
    'plugins.surface' : 'plugins.trisurface',
}

def find_global(module,name):
    """Override the import path of some classes"""
    print "I want to import %s from %s" % (name,module)
    if module in module_relocations:
        module = module_relocations[module]
        print "  I will try module %s instead" % module
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


class Project(dict):
    """Project: a persistent storage of pyFormex data.

    A pyFormex Project is a regular Python dict that can contain named data
    of any kind, and can be saved to a file to create persistence over
    different pyFormex sessions.

    While the Project class is mainly used by pyFormex for the pyformex.PF
    global variable that collects variables exported from pyFormex scripts,
    the user may also create and handle his own Project objects.

    .. warning:: In the current implementation there is no guarantee that
      a project file written by one version of pyFormex can be read back
      by a later version. The project files are mainly intended as a means
      to easily save lots of data and to restore them in your next session,
      or pass them to another user (with the same pyFormex version). If you
      need guarantee that you stored data will always remain accessible,
      you can save geometry type of data to pyFormex .pgf format files.
      Other data you should take care of yourself.

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
      the file. The 'r' access mode is a read-only mode. 
      ======  ===============  ============  ===================
      access  File must exist  File is read  File can be written
      ======  ===============  ============  ===================
        r           yes             yes             no
        rw          yes             yes             yes
        wr          no         if it exists         yes  
        w           no              no              yes 
      ======  ===============  ============  ===================
         
    - `signature`: A text that will be written in the header record of the
      file. This can e.g. be used to record format version info.

    - `compression`: An integer from 0 to 9: compression level. For large
      data sets, compression leads to much smaller files. 0 is no compression,
      9 is maximal compression. The default is 4.

    - `binary`: if False and no compression is used, storage is done
      in an ASCII format, allowing to edit the file. Otherwise, storage
      uses a binary format. Using binary=False is deprecated though.

    - `legacy`: if True, the Project is allowed to read old headerless file
      formats. This option has no effect on the creation of new Project files.

    - `data`: a dict-like object to initialize the Project contents. These data
      may override values read from the file.  
    """

    def __init__(self,filename=None,access='wr',signature=_signature_,compression=5,binary=True,data={},**kargs):
        """Create a new project."""
        if 'create' in kargs:
            import warnings
            warnings.warn("The create=True argument should be replaced with access='w'")
        if 'legacy' in kargs:
            import warnings
            warnings.warn("The legacy=True argument has become superfluous")
            
        self.filename = filename
        self.access = access 
        self.signature = str(signature)
        self.gzip = compression if compression in range(1,10) else 0
        self.mode = 'b' if binary else ''
        
        dict.__init__(self)
        if filename and os.path.exists(filename) and 'r' in self.access:
            # read existing contents
            self.load()
        self.update(data)
        if filename and access=='w':
            # destroy existing contents
            self.save()


    def header_data(self):
        """Construct the data to be saved in the header."""
        store_attr = ['signature','gzip','mode','autofile','_autoscript_']
        store_vals = [getattr(self,k,None) for k in store_attr]
        return dict([(k,v) for k,v in zip(store_attr,store_vals) if v is not None])


    def save(self):
        """Save the project to file."""
        if 'w' not in self.access:
            #print "NOT saving to readonly Project file access" 
            return

        if self.filename is None:
            import tempfile
            fd,fn = tempfile.mkstemp(prefix='pyformex_',suffix='.pyf')
            self.filename = fn
        else:
            print "Saving project %s with compression %s" % (self.filename,self.gzip)
        f = open(self.filename,'w'+self.mode)
        # write header
        f.write("%s\n" % self.header_data())
        f.flush()
        if self.mode == 'b':
            # When using binary, can as well use highest protocol
            protocol = cPickle.HIGHEST_PROTOCOL
        else:
            protocol = 0
        if self.gzip:
            pyf = gzip.GzipFile(self.filename,'w'+self.mode,self.gzip,f)
            cPickle.dump(self,pyf,protocol)
            pyf.close()
        else:
            cPickle.dump(self,f,protocol)
        f.close()


    def readHeader(self):
        """Read the header from a project file.
        
        Tries to read the header from different legacy formats,
        and if succesfull, adjusts the project attributes according
        to the values in the header.
        Returns the open file if succesfull.
        """
        self.format = -1
        print("Reading project file: %s" % self.filename)
        f = open(self.filename,'rb')
        s = f.readline()
        print "header = %s" % s
        # Try subsequent formats
        try:
            # newest format has header in text format
            header = eval(s)
            self.__dict__.update(header)
            self.format = 3
        except:

            # try OLD new format: the first pickle contains attributes
            try:
                print "trying format 2"
                p = pickle_load(f)
                self.__dict__.update(p)
                self.format = 2
            except:
                pass

            if 'gzip' in s:
                # transitional format
                self.gzip = 5
                self.format = 1
            else:
                # headerless format
                self.legacy = True
                f.seek(0)
                self.gzip = 0
                self.format = 0
                
            import warnings
            warnings.warn("This is an old format project file. Unless you need to read this project file from an older pyFormex version, we strongly advise you to convert the project file to the latest format. Otherwise future versions of pyFormex might not be able to read it back.")

        return f


    def load(self,try_resolve=False):
        """Load a project from file.
        
        The loaded definitions will update the current project.
        """
        f = self.readHeader()
        if f:
            if self.gzip:
                pyf = gzip.GzipFile(self.filename,'r',self.gzip,f)
                p = pickle_load(pyf,try_resolve)
                pyf.close()
            else:
                p = pickle_load(f,try_resolve)
            self.update(p)
        f.close()
    

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
        print "GOT KEYS %s" % self.keys()
        if filename is not None:
            self.filename = filename
        self.access = 'w'
        print "Will now save to %s" % self.filename
        self.save()
    

    def uncompress(self):
        """Uncompress a compressed project file.
        
        The project file is read, and if successfull, is written
        back in uncompressed format. This allows to make conversions
        of the data inside.
        """
        f = self.readHeader()
        print self.format,self.gzip
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
                print "Uncompressed %s to %s" % (self.filename,fn)
                
            else:    
                import warnings
                warnings.warn("The contents of the file does not appear to be compressed.")
            f.close()

 
    def delete(self):
        """Unrecoverably delete the project file."""
        os.remove(self.filename)
        
        
# Test

if __name__ == '__main__':

    d = dict(a=1,b=2,c=3,d=[1,2,3],e={'f':4,'g':5})
    from numpy import random
    d['r'] = random.randint(0,100,(3,3))
    print('DATA',d)

    P = Project('testa.pyf')
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)

    P = Project('testb.pyf',access='w')
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)

    P = Project()
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)

    for i in [0,1,3,5,7,9]:
        P = Project('testc%s.pyf'%i,access='w',compression=i)
        P.update(d)
        print('SAVE',P)
        P.save()
        P.clear()
        print('CLEAR',P)
        P.load()
        print('LOAD',P)
     
    P = Project('testl.pyf',access='w',legacy=True)
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)
     
    P = Project('testr.pyf',access='r')
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)

    #

# End
