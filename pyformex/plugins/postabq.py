#!/usr/bin/env python pyformex
# $Id$

"""A postprocessor for ABAQUS output files.

The ABAQUS .fil output file can be scanned and translated into a pyFormex
script with the 'postabq' command. Use it as follows:
  postabq job.fil > job.py

Then execute the created script from inside pyFormex.
"""

from numpy import *
import connectivity

class Database(object):
    def __init__(self):
        self.modeldone = False
        self.labels = {}
        self.nelems = 0
        self.nnodes = 0
        self.nodid = None
        self.nodes = None
        self.elems = None
       


nodnr = 0
elnr = 0

DB = Database()

def Initialize():
    global DB,nodnr,elnr
    Database.__init__(DB)
    nodnr = 0
    elnr = 0

def do_nothing(*arg,**kargs):
    """A do nothing function to stand in for all not yet defined functions."""
    pass

Abqver = do_nothing
Date = do_nothing
Elemset = do_nothing
Nodeset = do_nothing
Dofs = do_nothing
Heading = do_nothing
Increment = do_nothing
EndIncrement = do_nothing
TotalEnergies = do_nothing
OutputRequest = do_nothing
Coordinates = do_nothing
Displacements = do_nothing
NodeOutput = do_nothing
ElemHeader = do_nothing
ElemOutput = do_nothing
Unknown = do_nothing

def Size(nelems=10,nnodes=11,length=2.000000):
    DB.nelems = nelems
    DB.nnodes = nnodes
    DB.length = length
    DB.nodid = -ones((nnodes,),dtype=int32)
    DB.nodes = zeros((nnodes,3),dtype=float32)
    DB.elems = {}

def Node(nr,coords,normal=None):
    global nodnr
    DB.nodid[nodnr] = nr
    nn = len(coords)
    DB.nodes[nodnr][:nn] = coords
    nodnr += 1

def Element(nr,typ,conn):
    if not DB.elems.has_key(typ):
        DB.elems[typ] = []
    DB.elems[typ].append(conn)

def Finalize():
    DB.nid = connectivity.reverseIndex(DB.nodid.reshape((-1,1))).ravel()
    for k,v in DB.elems.iteritems():
        DB.elems[k] = array(v) - 1
    DB.modeldone = True

def Increment(*args,**kargs):
    if not DB.modeldone:
        Finalize()

def Label(tag,value):
    DB.labels[tag] = value
#End
