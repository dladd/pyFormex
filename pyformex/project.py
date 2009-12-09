# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""project.py

Functions for managing a project in pyFormex.
"""

import os
import cPickle as pickle
import gzip

_signature_ = 'pyFormex'

class Project(dict):
    """A project is a persistent storage of a Python dictionary."""

    def __init__(self,filename,create=False,signature=_signature_,compression=0,binary=False,legacy=True):
        """Create a new project with the given filename.

        If the filename exists and create is False, the file is opened and
        the contents is read into the project dictionary.
        If not, a new empty file and project are created.

        If legacy = True, the Project is allowed to read unsigned file formats.
        Writing is always done with signature though.
        """
        dict.__init__(self)
        self.filename = filename
        self.signature = signature
        if not compression in range(1,10):
            compression = 0
        self.gzip = compression
        if binary:
            self.mode = 'b'
        else:
            self.mode = ''
        self.legacy=legacy
        if create or not os.path.exists(filename):
            self.save()
        else:
            self.load()


    def header_data(self):
        """Construct the data to be saved in the header."""
        store_attr = ['gzip','mode','autofile']
        store_vals = [getattr(self,k,None) for k in store_attr]
        return dict([(k,v) for k,v in zip(store_attr,store_vals) if v is not None])


    def set_data_from_header(self,data):
        """Set the project data from the header."""
        d = eval(data)
        self.__dict__.update(d)


    def save(self):
        """Save the project to file."""
        f = file(self.filename,'w'+self.mode)
        f.write("%s\n" % self.signature)
        pickle.dump(self.header_data(),f,pickle.HIGHEST_PROTOCOL)
        if self.gzip:
            pyf = gzip.GzipFile(self.filename,'w'+self.mode,self.gzip,f)
            pickle.dump(self,pyf,pickle.HIGHEST_PROTOCOL)
            pyf.close()
        else:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        f.close()


    def load(self):
        """Load a project from file.
        
        The loaded definitions will update the current project.
        """
        f = file(self.filename,'rb')
        s = f.readline()
        if s.startswith(self.signature):
            # This is the new format: the first pickle contains attributes
            p = pickle.load(f)
            self.__dict__.update(p)
        elif 'gzip' in s:
            # Enables readback of transitionary format
            self.gzip = 5
        else:
            # Enable readback of legacy format
            if self.legacy:
                f.seek(0)
                self.gzip = 0
            else:
                # Incompatible format
                print("HEADER IS %s" % s)
                print("EXPECTED %s" % self.signature)
                raise ValueError,"File %s does not have a matching signature\nIf it is an old project file, try opening it with the 'legacy' option checked." % self.filename
        if self.gzip:
            pyf = gzip.GzipFile(self.filename,'r',self.gzip,f)
            p = pickle.load(pyf)
            pyf.close()
        else:
            p = pickle.load(f)
        f.close()
        self.update(p)

 
    def delete(self):
        """Unrecoverably delete the project file."""
        os.remove(self.filename)
        
        
# Test

if __name__ == '__main__':

    d = dict(a=1,b=2,c=3,d=[1,2,3],e={'f':4,'g':5})
    print('DATA',d)
    P = Project('test.pyf',create=True,signature='Test project')
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)
    
    P = Project('testc.pyf',create=True,signature='Test project',legacy=True)
    P.update(d)
    print('SAVE',P)
    P.save()
    P.clear()
    print('CLEAR',P)
    P.load()
    print('LOAD',P)

    #

# End
