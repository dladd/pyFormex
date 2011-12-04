# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
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

"""A postprocessor for ABAQUS output files.

The ABAQUS .fil output file can be scanned and translated into a pyFormex
script with the 'postabq' command. Use it as follows::

   postabq job.fil > job.py

Then execute the created script from inside pyFormex.
"""

import pyformex as pf
from arraytools import *
from script import export
from odict import ODict

import re

class FeResult(object):

    _name_ = '__FePost__'
    re_Skey = re.compile("S[0-5]")
    re_Ukey = re.compile("U[0-2]")

    def __init__(self,name=_name_,datasize={'U':3,'S':6,'COORD':3}):
        self.name = name
        self.datasize = datasize.copy()
        self.about = {'creator':pf.Version,
                      'created':pf.StartTime,
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

    def dataSize(self,key,data):
        if self.datasize.has_key(key):
            return self.datasize[key]
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
            self.datasize['U'] = 6
        
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
        self.nid = inverseUniqueIndex(self.nodid)
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
            self.R[key] = zeros((self.nnodes,self.dataSize(key,data)),dtype=float32)
        if key == 'U':
            self.R[key][nodid-1][self.displ-1] = data
        elif key == 'S':
            n1 = self.hdr['ndi']
            n2 = self.hdr['nshr']
            ind = arange(len(data))
            ind[n1:] += (3-n1)
            #print(ind)
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
        export({self.name:self, self._name_:self})
        pf.message("Read %d nodes, %d elements" % (self.nnodes,self.nelems))
        if self.res is None:
            pf.message("No results")
        else:
            pf.message("Steps: %s" % self.res.keys())

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


    def getSteps(self):
        """Return all the step keys."""
        return self.res.keys()

    def getIncs(self,step):
        """Return all the incs for given step."""
        if self.res.has_key(step):
            return self.res[step].keys()
        

    def getres(self,key,domain='nodes'):
        """Return the results of the current step/inc for given key.

        The key may include a component to return only a single column
        of a multicolumn value.
        """
        components = '012'
        if self.re_Skey.match(key):
            if self.datasize['S']==3:
                components = '013'
            else:
                components = '012345'
        elif self.re_Ukey.match(key):
            if self.datasize['U']==2:
                components = '01'
            else:
                components = '012'
        comp = components.find(key[-1])
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

        
    def printSteps(self):
        """Print the steps/increments/resultcodes for which we have results."""
        if self.res is not None:
            for i,step in self.res.iteritems():
                for j,inc in step.iteritems():
                    for k,v in inc.iteritems():
                        print("Step %s, Inc %s, Res %s (%s)" % (i,j,k,str(v.shape)))


#End
