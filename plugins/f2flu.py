#!/usr/local/bin/pyformex
#
"""Formex to Fluent translator.

This module contains some functions that can aid in exporting
pyFormex models to Fluent.

This script should be executed with the command
   pyformex --nogui f2flu.py  <stl_model>
"""

import sys
from plugins import tetgen
from plugins.elements import Tet4 


def read_tetgen(filename):
    """Read a tetgen tetraeder model.

    filename is the base of the path of the input files.
    For a filename 'proj', nodes are expected in 'proj.1.node'  and
    elems are in file 'proj.1.ele'.
    """
    nodes = tetgen.readNodes(filename+'.1.node')
    print "Read %d nodes" % nodes.shape[0]
    elems = tetgen.readElems(filename+'.1.ele')
    print "Read %d tetraeders" % elems.shape[0]
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
    print "Nodal coordinates"
    print nodes
    print "Element connectivity"
    print elems
    faces = array(Tet4.faces)   # Turning faces into an array is important !
    print "Tetraeder faces"
    print faces
    elf = elems.take(faces,axis=1)
    # Remark: the shorter syntax elems[faces] takes its elements along the
    #         axis 0. Then we would need to transpose() first (and probably
    #         swap axes again later)
    print "The faces of the elements:"
    print elf
    # We need a copy to sort the nodes (sorting is done in-place)
    elfs = elf.copy()
    elfs.sort(axis=2) 
    print "The faces with sorted nodes:"
    print elfs
    magic = elems.max()+1
    print "Magic number = %d" % magic
    code = encode(elfs[:,:,0],elfs[:,:,1],elfs[:,:,2],magic)
    # Remark how nice the encode function works on the whole array
    print "Encoded faces:"
    print code
    code = code.ravel()
    print code
    print "Just A Check:"
    print "Element 5 face 2 is %s " % elf[5,2]
    print "Element 5 face 2 is %s " % list(decode(code[4*5+2],magic))
    srt = code.argsort()
    print srt
    print code[srt]
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
            print "Face %s belongs to el %s and el %s" % ( elf[e1,f1], e2, e1 )

    
def tetgen2fluent(filename):
    """Convert a tetgen tetraeder model to fluent.

    filename is the base path of the tetgen input files.
    This will create a Fluent model in filename+'.flu'
    """
    nodes,elems = read_tetgen(filename)
    if nodes is None or elems is None:
        print "Error while reading model %s" % filename
        return
    fil = file(filename+'.flu','w')
    if fil:
        output_fluent(fil,nodes,elems)
        fil.close()

# This is special for pyFormex scripts !
if __name__ == "script": 

    for arg in argv:
        print "Converting model %s" % arg
        tetgen2fluent(arg)

    argv = [ 'hallo' ]


# End
