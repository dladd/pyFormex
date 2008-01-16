#!/usr/bin/env python


"""Centerline.py

Determine the (inner) voronoi diagram of a triangulated surface.
Determine approximation for the centerline.
"""

import os
from numpy import *
from plugins import surface,tetgen
from utils import runCommand


def det3(f):
    """Calculate the determinant of each of the 3 by 3 arrays.
    
    f is a (n,3,3) array.
    The output is 1d array containing the n corresponding determinants.
    """
    det = f[:,0,0]*(f[:,1,1]*f[:,2,2]-f[:,1,2]*f[:,2,1]) - f[:,0,1]*(f[:,1,0]*f[:,2,2]-f[:,1,2]*f[:,2,0]) + f[:,0,2]*(f[:,1,0]*f[:,2,1]-f[:,1,1]*f[:,2,0])
    return det


def det4(f):
    """Calculate the determinant of each of the 4 by 4 arrays.
    
    f is a (n,4,4) array.
    The output is 1d array containing the n corresponding determinants.
    """
    det = f[:,0,0]*det3(f[:,1:,1:])-f[:,0,1]*det3(f[:,1:,[0,2,3]])+f[:,0,2]*det3(f[:,1:,[0,1,3]])-f[:,0,3]*det3(f[:,1:,[0,1,2]])
    return det


def circumcenter(nodes,elems):
    """Calculate the circumcenters of a list of tetrahedrons.
    
    For a description of the method: http://mathworld.wolfram.com/Circumsphere.html
    The output are the circumcenters and the corresponding radii.
    """
    kwadSum = (nodes*nodes).sum(axis=-1)
    one = zeros(nodes.shape[0])+1
    nodesOne = append(nodes,one.reshape(-1,1),1)
    nodesKwadOne = append(kwadSum.reshape(-1,1),nodesOne,1)
    #construct necessary 4 by 4 arrays
    Wx = nodesKwadOne[:,[0,2,3,4]]
    Wy = nodesKwadOne[:,[0,1,3,4]]
    Wz = nodesKwadOne[:,[0,1,2,4]]
    #calculate derminants of the 4 by 4 arrays
    Dx = det4(Wx[elems])
    Dy = -det4(Wy[elems])
    Dz = det4(Wz[elems])
    alfa = det4(nodesOne[elems[:]])
    #circumcenters
    centers = column_stack([Dx[:]/(2*alfa[:]),Dy[:]/(2*alfa[:]),Dz[:]/(2*alfa[:])])
    #calculate radii of the circumscribed spheres
    vec = centers[:]-nodes[elems[:,0]]
    rad = sqrt((vec*vec).sum(axis=-1))
    return centers,rad

## Voronoi: vor diagram is determined using Tetgen. Some of the vor nodes may fall outside the surface. This should be avoided as this may compromise the centerline determination. Therefore, we created a second definition to determine the inner voronoi diagram (voronoiInner).

def voronoi(fn):
    """Determine the voronoi diagram corresponding with a triangulated surface.
    
    fn is the file name of a surface, including the extension (.off, .stl, .gts, .neu or .smesh)
    The voronoi diagram is determined by Tetgen.
    The output are the voronoi nodes and the corresponding radii of the voronoi spheres.
    """
    S = surface.Surface.read(fn)
    fn,ftype = os.path.splitext(fn)
    ftype = ftype.strip('.').lower()
    if ftype != 'smesh':
        S.write('%s.smesh' %fn)
    sta,out = runCommand('tetgen -zpv %s.smesh' %fn)
    #information tetrahedra
    elems = tetgen.readElems('%s.1.ele' %fn)[0]
    nodes = tetgen.readNodes('%s.1.node' %fn)[0]
    #voronoi information
    nodesVor = tetgen.readNodes('%s.1.v.node' %fn)[0]
    #calculate the radii of the voronoi spheres
    vec = nodesVor[:]-nodes[elems[:,0]]
    rad = sqrt((vec*vec).sum(axis=-1))
    return nodesVor,rad


def voronoiInner(fn):
    """Determine the inner voronoi diagram corresponding with a triangulated surface.
    
    fn is the file name of a surface, including the extension (.off, .stl, .gts, .neu or .smesh)
    The output are the voronoi nodes and the corresponding radii of the voronoi spheres.
    """
    S = surface.Surface.read(fn)
    fn,ftype = os.path.splitext(fn)
    ftype = ftype.strip('.').lower()
    if ftype != 'smesh':
        S.write('%s.smesh' %fn)
    sta,out = runCommand('tetgen -zp %s.smesh' %fn)
    #information tetrahedra
    elems = tetgen.readElems('%s.1.ele' %fn)[0]
    print elems.shape
    nodes = tetgen.readNodes('%s.1.node' %fn)[0]
    print nodes.shape
    #calculate surface normal for each point
    elemsS = array(S.elems)
    NT = S.areaNormals()[1]
    NP = zeros([nodes.shape[0],3])
    for i in [0,1,2]:
        NP[elemsS[:,i]] = NT
    #calculate centrum circumsphere of each tetrahedron
    centers = circumcenter(nodes,elems)[0]
    #check if circumcenter falls within the geomety described by the surface
    ie = column_stack([((nodes[elems[:,j]] - centers[:])*NP[elems[:,j]]).sum(axis=-1) for j in [0,1,2,3]])
    ie = ie[:,:]>=0
    w = where(ie.all(1))[0]
    elemsInner = elems[w]
    nodesVorInner = centers[w]
    #calculate the radii of the voronoi spheres
    vec = nodesVorInner[:]-nodes[elemsInner[:,0]]
    rad = sqrt((vec*vec).sum(axis=-1))
    return nodesVorInner,rad
    
    
def selectMaxVor(nodesVor,rad,r1=1.,r2=2.,q=0.7,maxruns=-1):
    """Select the local maxima of the voronoi spheres.
    
    Description of the procedure:
    1) The largest voronoi sphere in the record is selected (voronoi node N and radius R).
    2) All the voronoi nodes laying within a cube all deleted from the record.
    This cube is defined by:
        a) the centrum of the cube N.
        b) the edge length which is 2*r1*R.
    3) Some voronoi nodes laying within a 2nd, larger cube are also deleted.
    This is when their corresponding radius is smaller than q times R.
    This cube is defined by:
        a) the centrum of the cube N.
        b) the edge length which is 2*r2*R.
    4) These three operations are repeated until all nodes are deleted.
    """
    nodesCent = array([])   
    radCent = array([]) 
    run = 0
    while nodesVor.shape[0] and (maxruns < 0 or run < maxruns):
        #find maximum voronoi sphere in the record
        w = where(rad[:] == rad[:].max())[0]
        maxR = rad[w].reshape(-1)
        maxP = nodesVor[w].reshape(-1)
        #remove all the nodes within the first cube
        t1 =  (nodesVor[:] > (maxP-r1*maxR)).all(axis=1)
        t2 =  (nodesVor[:] < (maxP+r1*maxR)).all(axis=1)
        ttot1 = t1*t2
        rad = rad[-ttot1]
        nodesVor = nodesVor[-ttot1]
        #remove some of the nodes within the second cube
        t3 =  (nodesVor[:] > (maxP-r2*maxR)).all(axis=1)
        t4 =  (nodesVor[:] < (maxP+r2*maxR)).all(axis=1)
        t5 = (rad<maxR*q)
        ttot2 = t3*t4*t5
        if ttot2.shape[0]:
            rad = rad[-ttot2]
            nodesVor = nodesVor[-ttot2]
        #add local maximum to a list
        nodesCent = append(nodesCent,maxP)
        radCent = append(radCent,maxR)
        run += 1
    return nodesCent.reshape(-1,1,3),radCent
