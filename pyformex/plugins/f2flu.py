# $Id$      *** pyformex ***
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""Formex to Fluent translator.

This module contains some functions that can aid in exporting
pyFormex models to Fluent.

This script should be executed with the command
   pyformex --nogui f2flu.py  <stl_model>
"""

import sys
from plugins import tetgen
from elements import Tet4 
from time import strftime, gmtime
from numpy import *

def writeHeading(fil, nodes, elems, text=''):
    """Write the heading of the Gambit neutral file.""" #currently only for hexahedral mesh
    fil.write("        CONTROL INFO 2.2.30\n")
    fil.write("** GAMBIT NEUTRAL FILE\n")
    fil.write('%s\n' %text)
    fil.write('PROGRAM:                Gambit     VERSION:  2.2.30\n')
    fil.write(strftime('%d %b %Y    %H:%M:%S\n', gmtime()))
    fil.write('     NUMNP     NELEM     NGRPS    NBSETS     NDFCD     NDFVL\n')
    fil.write('%10i%10i%10i%10i%10i%10i\n' % (shape(nodes)[0],shape(elems)[0],1,0,3,3))
    fil.write('ENDOFSECTION\n')

def writeNodes(fil, nodes, nofs=1):
    """Write nodal coordinates.

    The nofs specifies an offset for the node numbers.
    The default is 1, because Gambit numbering starts at 1.  
    """
    fil.write('   NODAL COORDINATES 2.2.30\n')
    for i,n in enumerate(nodes):
        fil.write("%10d%20.11e%20.11e%20.11e\n" % ((i+nofs,)+tuple(n)))
    fil.write('ENDOFSECTION\n')

def writeElems(fil, elems1, eofs=1, nofs=1):
    """Write element connectivity.

    The eofs and nofs specify offsets for element and node numbers.
    The default is 1, because Gambit numbering starts at 1.  
    """
    #pyFormex uses the same convention for hexahedral elements as ABAQUS
    #Gambit uses a different convention
    #function currently only for hexahedral mesh
    elems = elems1.copy()
    elems[:,2] = elems1[:,3]
    elems[:,3] = elems1[:,2]

    elems[:,6] = elems1[:,7]
    elems[:,7] = elems1[:,6]
    
    fil.write('      ELEMENTS/CELLS 2.2.30\n')
    for i,e in enumerate(elems+nofs):
        fil.write('%8d %2d %2d %8d%8d%8d%8d%8d%8d%8d\n               %8d\n' % ((i+eofs,4,8)+tuple(e)))
    fil.write('ENDOFSECTION\n')
    
def writeGroup(fil, elems):
    """Write group of elements.

    The eofs and nofs specify offsets for element and node numbers.
    The default is 1, because Gambit numbering starts at 1.  
    """
    fil.write('       ELEMENT GROUP 2.2.30\n')
    fil.write('GROUP:%11d ELEMENTS:%11d MATERIAL:%11d NFLAGS:%11d\n' % (1,shape(elems)[0],2,1))
    fil.write('%32s\n' %'fluid')
    fil.write('%8d\n' %0)
    n =  shape(elems)[0]/10
    for i in range(n):
        fil.write('%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n' %(10*i+1,10*i+2,10*i+3,10*i+4,10*i+5,10*i+6,10*i+7,10*i+8,10*i+9,10*i+10))
    for j in range(shape(elems)[0]-10*n):
        fil.write('%8d' %(10*n+j+1))
    fil.write('\n')
    fil.write('ENDOFSECTION\n')

def read_tetgen(filename):
    """Read a tetgen tetraeder model.

    filename is the base of the path of the input files.
    For a filename 'proj', nodes are expected in 'proj.1.node'  and
    elems are in file 'proj.1.ele'.
    """
    nodes = tetgen.readNodes(filename+'.1.node')
    print("Read %d nodes" % nodes.shape[0])
    elems = tetgen.readElems(filename+'.1.ele')
    print("Read %d tetraeders" % elems.shape[0])
    return nodes,elems


def encode(i,j,k,n):
    return n*(n*i+j)+k

def decode(code,n):
    q,k = code/n, code%n
    i,j = q/n, q%n
    return i,j,k

def output_fluent(fil,nodes,elems):
    """Write a tetraeder mesh in Fluent format to fil.

    The tetraeder mesh consists of an array of nodal coordinates
    and an array of element connectivity.
    """
    print("Nodal coordinates")
    print(nodes)
    print("Element connectivity")
    print(elems)
    faces = array(Tet4.faces[1])   # Turning faces into an array is important !
    print("Tetraeder faces")
    print(faces)
    elf = elems.take(faces,axis=1)
    # Remark: the shorter syntax elems[faces] takes its elements along the
    #         axis 0. Then we would need to transpose() first (and probably
    #         swap axes again later)
    print("The faces of the elements:")
    print(elf)
    # We need a copy to sort the nodes (sorting is done in-place)
    elfs = elf.copy()
    elfs.sort(axis=2) 
    print("The faces with sorted nodes:")
    print(elfs)
    magic = elems.max()+1
    print("Magic number = %d" % magic)
    code = encode(elfs[:,:,0],elfs[:,:,1],elfs[:,:,2],magic)
    # Remark how nice the encode function works on the whole array
    print("Encoded faces:")
    print(code)
    code = code.ravel()
    print(code)
    print("Just A Check:")
    print("Element 5 face 2 is %s " % elf[5,2])
    print("Element 5 face 2 is %s " % list(decode(code[4*5+2],magic)))
    srt = code.argsort()
    print(srt)
    print(code[srt])
    # Now shipout the faces in this order, removing the doubles
    j = -1 
    for i in srt:
        if j < 0: # no predecessor (or predecessor already shipped)
            j = i
        else:
            e1,f1 = j/4, j%4
            if code[i] == code[j]:
                e2,f2 = i/4, i%4
                j = -1
            else:
                e2 = -1
                j = i
            print("Face %s belongs to el %s and el %s" % ( elf[e1,f1], e2, e1 ))

    
def tetgen2fluent(filename):
    """Convert a tetgen tetraeder model to fluent.

    filename is the base path of the tetgen input files.
    This will create a Fluent model in filename+'.flu'
    """
    nodes,elems = read_tetgen(filename)
    if nodes is None or elems is None:
        print("Error while reading model %s" % filename)
        return
    fil = open(filename+'.flu','w')
    if fil:
        output_fluent(fil,nodes,elems)
        fil.close()

# This is special for pyFormex scripts !
if __name__ == "script": 

    for arg in argv:
        print("Converting model %s" % arg)
        tetgen2fluent(arg)

    argv = [ 'hallo' ]


# End
