# $Id$

"""project.py

Functions for managing a project in pyFormex.
"""

import os
import cPickle as pickle

class Project(dict):
    """A pyFormex project is a persistent storage of pyFormex objects."""

    def __init__(self,filename,create=False):
        """Create a new project with the given filename.

        If the filename exists, it is opened and the contents is read.
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
