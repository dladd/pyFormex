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

class Project(dict):
    """A pyFormex project is a persistent storage of pyFormex objects."""

    def __init__(self,filename,create=False):
        """Create a new project with the given filename.

        If the filename exists and create is False, the file is opened and
        the contents is read into the project dictionary.
        If not, a new empty file and project are created.
        """
        dict.__init__(self)
        self.filename = filename
        if create or not os.path.exists(filename):
            self.save(filename)
        else:
            self.load(filename)

    def save(self,filename=None):
        """Save the project to file."""
        if filename is None:
            filename = self.filename
        f = file(filename,'w')
        pickle.dump(self,f)
        f.close()

    def load(self,filename=None):
        """Load a project from file.
        
        The loaded definition will update the current project.
        """
        if filename is None:
            filename = self.filename
        f = file(filename,'r')
        p = pickle.load(f)
        f.close()
        if isinstance(p,dict):
            self.update(p)
 
    def delete(self):
        """Unrecoverably delete the project file."""
        os.remove(self.filename)
        
        
# Test

if __name__ == '__main__':

    d = dict(a=1,b=2,c=3,d=[1,2,3],e={'f':4,'g':5})
    print d
    P = Project('d.test',create=True)
    P.update(d)
    print P
    P.save()
    P.clear()
    print P
    P.load()
    print P
    

# End
