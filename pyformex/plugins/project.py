# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""project.py

Functions for managing a project in pyFormex.
"""

import os
import cPickle as pickle
import gzip

_signature_ = 'project'

class Project(dict):
    """A pyFormex project is a persistent storage of pyFormex objects."""

    def __init__(self,filename,create=False,compressed=False,signature=_signature_,legacy=False):
        """Create a new project with the given filename.

        If the filename exists and create is False, the file is opened and
        the contents is read into the project dictionary.
        If not, a new empty file and project are created.

        If legacy = True, the Project is allowed to read unsigned file formats.
        Writing is always done with signature though.
        """
        dict.__init__(self)
        self.filename = filename
        self.compressed = compressed
        self.signature = signature
        self.legacy = legacy
        if create or not os.path.exists(filename):
            self.save()
        else:
            self.load()


    def save(self,filename=None,compressed=None,signature=None):
        """Save the project to file."""
        if filename is None:
            filename = self.filename
        if compressed is None:
            compressed = self.compressed
        if signature is None:
            signature = self.signature

        print self.compressed,compressed
        f = file(filename,'wb')
        if compressed:
            if not type(compressed) is int and compressed in range(1,10):
                compressed = 5
            f.write('%s gzip %s\n'%(self.signature,compressed))
            pyf = gzip.GzipFile(filename,'wb',compressed,f)
        else:
            if not self.legacy:
                f.write('%s\n'%(self.signature))
            pyf = f
        self.compressed = compressed
        pickle.dump(self,pyf,pickle.HIGHEST_PROTOCOL)
        if compressed:
            pyf.close()
        f.close()


    def load(self,filename=None,compressed=None,signature=None):
        """Load a project from file.
        
        The loaded definition will update the current project.
        """
        if filename is None:
            filename = self.filename
        if compressed is None:
            compressed = self.compressed
        if signature is None:
            signature = self.signature

        f = file(filename,'rb')
        s = f.readline()
        if s.startswith(self.signature):
            s = s.split()
            if s[-2] == 'gzip':
                compressed = int(s[-1])
        else:
            if self.legacy:
                f.seek(0)
                compressed = False
            else:
                raise ValueError,"File %s does not have a matching signature" % filename
        
        if compressed:
            pyf = gzip.GzipFile(filename,'rb',compressed,f)
        else:
            pyf = f
        p = pickle.load(pyf)
        if compressed:
            pyf.close()
        f.close()
        self.update(p)

 
    def delete(self):
        """Unrecoverably delete the project file."""
        os.remove(self.filename)
        
        
# Test

if __name__ == '__main__':

    d = dict(a=1,b=2,c=3,d=[1,2,3],e={'f':4,'g':5})
    print 'DATA',d
    P = Project('test.pyf',create=True,signature='Test project')
    P.update(d)
    print 'SAVE',P
    P.save()
    P.clear()
    print 'CLEAR',P
    P.load()
    print 'LOAD',P
    
    P = Project('testc.pyf',create=True,signature='Test project',compressed=True)
    P.update(d)
    print 'SAVE',P
    P.save(compressed=True)
    P.clear()
    print 'CLEAR',P
    P.load(compressed=True)
    print 'LOAD',P

    #

# End
