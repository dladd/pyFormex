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

DataSize = { 'U':3,
             'S':6,
             }

class FeResult(object):

    def __init__(self):
        self.about = {'creator':GD.Version,
                      'created':GD.date,
                      }
        self.modeldone = False
        self.labels = {}
        self.nelems = 0
        self.nnodes = 0
        self.dofs = None
        self.nodid = None
        self.nodes = None
        self.elems = None
        self.nset = None
        self.eset = None
        self.res = None
        self.hdr = None
        self.nodnr = 0
        self.elnr = 0

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
        self.nset[key] = asarray(data)

    def Elemset(self,key,data):
        self.eset[key] = asarray(data)

    def Finalize(self):
        self.nid = connectivity.reverseIndex(self.nodid.reshape((-1,1))).ravel()
        for k,v in self.elems.iteritems():
            self.elems[k] = array(v) - 1
        self.modeldone = True
        self.res = {}

    def Increment(self,step,inc,**kargs):
        if not self.modeldone:
            self.Finalize()
        if step != self.step:
            self.step = step
            if not self.res.has_key(self.step):
                self.res[self.step] = {}
            self.inc = 0
        res = self.res[self.step]
        if inc != self.inc:
            self.inc = inc
            if not res.has_key(self.inc):
                res[self.inc] = {}
        self.R = self.res[self.step][self.inc]
        
    def EndIncrement(self):
        self.step = self.inc = 0

    def Label(self,tag,value):
        self.labels[tag] = value

    def NodeOutput(self,key,nodid,data):
        if not self.R.has_key(key):
            self.R[key] = zeros((self.nnodes,DataSize[key]),dtype=float32)
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
        comp = '012345'.find(key[-1])
        if comp >= 0:
            key = key[:-1]
        if self.R.has_key(key):
            val = self.R[key]
            if comp >= 0:
                return val[:,comp]
            else:
                return val
        else:
            return None

#End
