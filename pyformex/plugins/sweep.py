#!/usr/bin/env python pyformex.py


"""Create a hexahedral mesh by sweeping a quadrilateral mesh along a path"""

from simple import line,circle
from numpy import *
from gui import actors


def sweepGrid(nodes,elems,path,scale=1.,angle=0.,a1=None,a2=None):
    """ Sweep a quadrilateral mesh along a path
    
    The path should be specified as a (n,2,3) Formex.
    The input grid (quadrilaterals) has to be specified with the nodes and elems and 
    can for example be created with the functions gridRectangle or gridBetween2Curves.
    This quadrilateral grid should be within the YZ-plane.
    The quadrilateral grid can be scaled and/or rotated along the path.
    
    There are three options for the first (a1) / last (a2) element of the path:
    1) None: No corresponding hexahedral elements
    2) 'last': The direction of the first/last element of the path is used to 
    direct the input grid at the start/end of the path
    3) specify a vector: This vector is used to direct the input grid at the start/end of the path
    
    The resulting hexahedral mesh is returned in terms of nodes and elems.
    """
    nodes = Formex(nodes.reshape(-1,1,3))
    n = nodes.shape()[0]
    s = path.shape()[0]
    sc = scale-1.
    a = angle
    
    if a1 != None:
        if a1 == 'last':
            nodes1 = nodes.rotate(actors.rotMatrix(path[0,1]-path[0,0])).translate(path[0,0])
        else:
            nodes1 = nodes.rotate(actors.rotMatrix(a1)).translate(path[0,0])
    else:
        nodes1 = Formex([[[0.,0.,0.]]])
    
    for i in range(s-1):
        r1 = vectorNormalize(path[i+1,1]-path[i+1,0])[1][0]
        r2 = vectorNormalize(path[i,1]-path[i,0])[1][0]
        r = r1+r2
        nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(r)).translate(path[i+1,0])
        scale = scale+sc
        angle = angle+a

    if a2 != None:    
        if a2 == 'last':
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(path[s-1,1]-path[s-1,0])).translate(path[s-1,1])
        else:
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(a2)).translate(path[s-1,1])
    
    if a1 == None:
        nodes1 = nodes1[1:]
        s = s-1
    if a2 == None:
        s = s-1

    elems0 = elems
    elems1 = append(elems0,elems+n,1)
    elems = elems1
    for i in range(s-1):
        elems = append(elems,elems1+(i+1)*n,0)
    if s == 0:
        elems = array([])
    
    return nodes1[:].reshape(-1,3),elems

### Some useful functions for creating quadrilateral meshes ############################    

def gridRectangle(n1,n2,width,height):
    """Create a rectangular grid with quadrilaterals
    
    Create a rectangular grid with n1 elems along the width
    and n2 elements along the height of the section.
    """
    sp = Formex([[[0,-width/2,-height/2]]])
    nodes = sp.replic(n1+1,width/n1,1).replic(n2+1,height/n2,2)
    elems = array([]).astype(int)
    for i in range(n1):
        for j in range(n2):
            elems = append(elems,[i+j*(n1+1),i+1+j*(n1+1),i+n1+2+j*(n1+1),i+n1+1+j*(n1+1)])
    return nodes[:],elems.reshape(-1,4)


def gridBetween2Curves(curve1,curve2,n):
    """Create a grid with quadrilaterals defined by two boundary curves
    
    The two curves should be (m,2,3) formices with the same number of elements!
    These curves should lay within the YZ plane!
    n is the number of elements between the two curves.
    """

    nc1 = append(curve1[:,0],curve1[-1,-1].reshape(-1,3),0)
    nc2 = append(curve2[:,0],curve2[-1,-1].reshape(-1,3),0)
    nodes = array([])
    elems = array([]).astype(int)
    for i in range(nc1.shape[0]):
        L = line(nc1[i],nc2[i],n)
        nL=append(L[:,0],L[-1,-1].reshape(-1,3),0)
        for j in nL:
            nodes = append(nodes,j)
    for i in range(n):
        for j in range(nc1.shape[0]-1):   
            elems = append(elems,[i+j*(n+1),i+1+j*(n+1),i+n+2+j*(n+1),i+n+1+j*(n+1)])
    return nodes.reshape(-1,3),elems.reshape(-1,4)
