# $Id$

"""Classes and functions for exporting geometry in various formats.


"""

import pyformex

class Mesh(object):
    def __init__(self,obj=None):
        self.coords = None
        self.elems = None
        if obj is not None:
            if hasattr(obj,'toMesh'):
                self.coords,self.elems = obj.toMesh()
            else:
                try:
                    self.coords,self.elems = obj
                except:
                    raise ValueError,"Invalid initialization data"
    

class ObjFile(object):
    def __init__(self,filename):
        self.file = file(filename,'w')
        self.file.write("# .obj file written by %s\n" % pyformex.Version)

    def write(self,mesh,name=None):

        if name is not None:
            self.file.write("o %s\n" % str(name))

        for v in mesh.coords:
            self.file.write("v %s %s %s\n" % tuple(v))

        # element code: p(oint), l(ine) or f(ace)
        nplex = mesh.elems.shape[1]
        code = { 1:'p', 2:'l' }.get(nplex,'f')
        s = code+(' %s'*nplex)+'\n'
        for e in mesh.elems+1:   # .obj format starts at 1
            self.file.write(s % tuple(e))

    def close(self):
        self.file.write('# End\n')
        self.file.close()
        self.file = None


# End
