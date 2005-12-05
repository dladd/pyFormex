#!/usr/bin/env pyformex
# $Id$
#
"""X-shaped truss analysis"""

from examples.X_truss import X_truss

# create a truss (Vandepitte, Chapter 1, p.16)
n = 5
l = 800.
h = 800.
truss = X_truss(n,l,h)

# draw it
clear()
draw(truss.allNodes())
draw(truss.allBars())

# assign property numbers
truss.bot.setProp(0)
truss.top.setProp(0)
truss.vert.setProp(2)
truss.mid1.setProp(1)
truss.mid2.setProp(1)
for p in [ truss.bot.p, truss.top.p ]:
    p[0] = p[n-1] = 3 

# define member properties
materials={ 'steel' : { 'E' : 207000, 'nu' : 0.3 } }
sections={ 'hor' : 50, 'end' : 40, 'dia' : 40, 'vert': 30 }
properties = { '0' : [ 'steel', 'hor' ],
               '3' : [ 'steel', 'end' ],
               '2' : [ 'steel', 'vert' ],
               '1' : [ 'steel', 'dia' ] }

def getmat(key):
    """Return the 'truss material' with key (str or int)."""
    p = properties.get(str(key),[None,None])
    m = materials.get(p[0],{})
    E = m.get('E',0.)
    rho = m.get('rho',0.)
    A = sections.get(p[1],0.)
    return [ E, rho, A ]


# create model for structural analysis
model = truss.allBars()
coords,elems = model.nodesAndElements()
props = model.prop()
propset = model.propSet()

clear()
draw(Formex(reshape(coords,(coords.shape[0],1,coords.shape[1]))))
draw(model)

# import analysis module
# You need calpy >= 0.2.1-pre1
# Download from ftp://bumps.ugent.be/calpy
import sys
sys.path.append('/usr/local/lib/calpy-0.2.1')
from fe_util import *
from truss3d import *
nnod = coords.shape[0]
nelems = elems.shape[0]
# boundary conditions
# we use the knowledge that the elements are in the order
# bot,top,vert,mid1,mid2
# remember to add 1 to number starting from 1, as needed by calpy
nr_fixed_support = elems[0][0]
nr_moving_support = elems[n-1][1]
nr_loaded = elems[2][1] # right node of the 3-d element
bcon = ReadBoundary(nnod,3,"""
  all  0  0  1
  %d   1  1  1
  %d   0  1  1
""" % (nr_fixed_support + 1,nr_moving_support + 1))
NumberEquations(bcon)
mats=array([ getmat(i) for i in range(max(propset)+1) ])
matnod = concatenate([reshape(props+1,(nelems,1)),elems+1],1)
ndof=bcon.max()
nlc=1
loads=zeros((ndof,nlc),Float)
loads[:,0]=AssembleVector(loads[:,0],[ 0.0, -50.0, 0.0 ],bcon[nr_loaded,:])
static(coords,bcon,mats,matnod,loads,Echo=True)
