# $Id$

"""A postprocessor for ABAQUS output files.

The ABAQUS .fil output file can be scanned and translated into a pyFormex
script with the 'postabq' command. Use it as follows:
  postabq job.fil > job.py

Then execute the created script from inside pyFormex.
"""

import globaldata as GD
from numpy import *
from gui.draw import export
import connectivity
from odict import ODict

class FeResult(object):

    data_size = {'U':3,'S':6,'COORD':3}

    def __init__(self):
        self.about = {'creator':GD.Version,
                      'created':GD.date,
                      }
        self.modeldone = False
        self.labels = {}
        self.nelems = 0
        self.nnodes = 0
        self.dofs = None
        self.displ = None
        self.nodid = None
        self.nodes = None
        self.elems = None
        self.nset = None
        self.nsetkey = None
        self.eset = None
        self.res = None
        self.hdr = None
        self.nodnr = 0
        self.elnr = 0

    def datasize(self,key,data):
        if FeResult.data_size.has_key(key):
            return FeResult.data_size[key]
        else:
            return len(data)

    def Abqver(self,version):
        self.about.update({'abqver':version})

    def Date(self,date,time):
        self.about.update({'abqdate':date,'abqtime':time})

    def Size(self,nelems,nnodes,length):
        self.nelems = nelems
        self.nnodes = nnodes
        self.length = length
        self.nodid = -ones((nnodes,),dtype=int32)
        self.nodes = zeros((nnodes,3),dtype=float32)
        self.elems = {}
        self.nset = {}
        self.eset = {}

    def Dofs(self,data):
        self.dofs = array(data)
        self.displ = self.dofs[self.dofs[:6] > 0]
        if self.displ.max() > 3:
            self.data_size['U'] = 6
        
    def Heading(self,head):
        self.about.update({'heading':head})

    def Node(self,nr,coords,normal=None):
        self.nodid[self.nodnr] = nr
        nn = len(coords)
        self.nodes[self.nodnr][:nn] = coords
        self.nodnr += 1

    def Element(self,nr,typ,conn):
        if not self.elems.has_key(typ):
            self.elems[typ] = []
        self.elems[typ].append(conn)

    def Nodeset(self,key,data):
        self.nsetkey = key
        self.nset[key] = asarray(data)

    def NodesetAdd(self,data):
        self.nset[self.nsetkey] = union1d(self.nset[self.nsetkey],asarray(data))

    def Elemset(self,key,data):
        self.esetkey = key
        self.eset[key] = asarray(data)

    def ElemsetAdd(self,data):
        self.eset[self.esetkey] = union1d(self.eset[self.esetkey],asarray(data))

    def Finalize(self):
        self.nid = connectivity.reverseUniqueIndex(self.nodid)
        for k in self.elems.iterkeys():
            v = asarray(self.elems[k])
            self.elems[k] = asarray(self.nid[v])
        self.modeldone = True
        # we use lists, to keep the cases in order
        self.res = ODict()
        self.step = None
        self.inc = None
 
    def Increment(self,step,inc,**kargs):
        if not self.modeldone:
            self.Finalize()
        if step != self.step:
            if step not in self.res.keys():
                self.res[step] = ODict()
            self.step = step
            self.inc = None
        res = self.res[self.step]
        if inc != self.inc:
            if inc not in res.keys():
                res[inc] = {}
            self.inc = inc
        self.R = self.res[self.step][self.inc]
        
    def EndIncrement(self):
        if not self.modeldone:
            self.Finalize()
        self.step = self.inc = -1

    def Label(self,tag,value):
        self.labels[tag] = value

    def NodeOutput(self,key,nodid,data):
        if not self.R.has_key(key):
            self.R[key] = zeros((self.nnodes,self.datasize(key,data)),dtype=float32)
        if key == 'U':
            self.R[key][nodid-1][self.displ-1] = data
        elif key == 'S':
            n1 = self.hdr['ndi']
            n2 = self.hdr['nshr']
            ind = arange(len(data))
            ind[n1:] += (3-n1)
            print ind
            self.R[key][nodid-1][ind] = data
        else:
            self.R[key][nodid-1][:len(data)] = data

    def ElemHeader(self,**kargs):
        self.hdr = dict(**kargs)

    def ElemOutput(self,key,data):
        if self.hdr['loc'] == 'na':
            self.NodeOutput(key,self.hdr['i'],data)

    def Export(self):
        """Align on the last increment and export results"""
        try:
            self.step = self.res.keys()[-1]
            self.inc = self.res[self.step].keys()[-1]
            self.R = self.res[self.step][self.inc]
        except:
            self.step = None
            self.inc = None
            self.R = None
        export({'DB':self})
        GD.message("Read %d nodes, %d elements" % (self.nnodes,self.nelems))
        if self.res is None:
            GD.message("No results")
        else:
            GD.message("Steps: %s" % self.res.keys())

    def do_nothing(*arg,**kargs):
        """A do nothing function to stand in for as yet undefined functions."""
        pass

    TotalEnergies = do_nothing
    OutputRequest = do_nothing
    Coordinates = do_nothing
    Displacements = do_nothing
    Unknown = do_nothing

    def setStepInc(self,step,inc):
        try:
            self.step = step
            self.inc = inc
            self.R = self.res[self.step][self.inc]
        except:
            self.R = {}


    def getres(self,key,domain='nodes'):
        """Return the results of the current step/inc for given key.

        The key may include a component to return only a single column
        of a multicolumn value.
        """
        print self.dofs
        print
        comp = '012345'.find(key[-1])
        if comp >= 0:
            key = key[:-1]
        if self.R.has_key(key):
            val = self.R[key]
            if comp in range(val.shape[1]):
                return val[:,comp]
            else:
                return val
        else:
            return None

#End
