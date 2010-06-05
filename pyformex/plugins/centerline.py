#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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


"""Centerline.py

Determine the (inner) voronoi diagram of a triangulated surface.
Determine approximation for the centerline.
"""

import pyformex as GD
import os
from numpy import *
from plugins import surface,tetgen
from utils import runCommand
import coords,connectivity


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


def encode2(i,j,n):
    return n*i+j

    
def decode2(code,n):
    i,j = code/n, code%n
    return i,j

    
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
    radii = sqrt((vec*vec).sum(axis=-1))
    return centers,radii

## Voronoi: vor diagram is determined using Tetgen. Some of the vor nodes may fall outside the surface. This should be avoided as this may compromise the centerline determination. Therefore, we created a second definition to determine the inner voronoi diagram (voronoiInner).

def voronoi(fn):
    """Determine the voronoi diagram corresponding with a triangulated surface.
    
    fn is the file name of a surface, including the extension (.off, .stl, .gts, .neu or .smesh)
    The voronoi diagram is determined by Tetgen.
    The output are the voronoi nodes and the corresponding radii of the voronoi spheres.
    """
    S = surface.TriSurface.read(fn)
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
    radii = sqrt((vec*vec).sum(axis=-1))
    return nodesVor,radii


def voronoiInner(fn):
    """Determine the inner voronoi diagram corresponding with a triangulated surface.
    
    fn is the file name of a surface, including the extension (.off, .stl, .gts, .neu or .smesh)
    The output are the voronoi nodes and the corresponding radii of the voronoi spheres.
    """
    S = surface.TriSurface.read(fn)
    fn,ftype = os.path.splitext(fn)
    ftype = ftype.strip('.').lower()
    if ftype != 'smesh':
        S.write('%s.smesh' %fn)
    sta,out = runCommand('tetgen -zp %s.smesh' %fn)
    #information tetrahedra
    elems = tetgen.readElems('%s.1.ele' %fn)[0]
    nodes = tetgen.readNodes('%s.1.node' %fn)[0].astype(float64)
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
    radii = sqrt((vec*vec).sum(axis=-1))
    return nodesVorInner,radii

    
def selectMaxVor(nodesVor,radii,r1=1.,r2=2.,q=0.7,maxruns=-1):
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
        w = where(radii[:] == radii[:].max())[0]
        maxR = radii[w].reshape(-1)
        maxP = nodesVor[w].reshape(-1)
        #remove all the nodes within the first cube
        t1 =  (nodesVor[:] > (maxP-r1*maxR)).all(axis=1)
        t2 =  (nodesVor[:] < (maxP+r1*maxR)).all(axis=1)
        ttot1 = t1*t2
        radii = radii[-ttot1]
        nodesVor = nodesVor[-ttot1]
        #remove some of the nodes within the second cube
        t3 =  (nodesVor[:] > (maxP-r2*maxR)).all(axis=1)
        t4 =  (nodesVor[:] < (maxP+r2*maxR)).all(axis=1)
        t5 = (radii<maxR*q)
        ttot2 = t3*t4*t5
        if ttot2.shape[0]:
            radii = radii[-ttot2]
            nodesVor = nodesVor[-ttot2]
        #add local maximum to a list
        nodesCent = append(nodesCent,maxP)
        radCent = append(radCent,maxR)
        run += 1
    return nodesCent.reshape(-1,1,3),radCent


def removeDoubles(elems):
    """Remove the double lines from the centerline.
    
    This is a clean-up function for the centerline.
    Lines appearing twice in the centerline are removed by this function.
    Both input and output are the connectivity of the centerline.
    """
    elems.sort(1)
    magic = elems.shape[0]+1
    code = encode2(elems[:,0],elems[:,1],magic)
    r = unique(code.reshape(-1))
    elemsU = decode2(r,magic)
    return transpose(array(elemsU))


def connectVorNodes(nodes,radii):
    """Create connections between the voronoi nodes.
    
    Each of the nodes is connected with its closest neighbours.
    The input is an array of n nodes and an array of n corresponding radii.
    Two voronoi nodes are connected if the distance between these two nodes
    is smaller than the sum of their corresponding radii.
    The output is an array containing the connectivity information.
    """
    connections = array([]).astype(int)
    v = 4
    for i in range(nodes.shape[0]):
        t1 =  (nodes[:] > (nodes[i]-v*radii[i])).all(axis=2)
        t2 =  (nodes[:] < (nodes[i]+v*radii[i])).all(axis=2)
        t = t1*t2
        t[i] = False
        w1 = where(t == 1)[0]
        c = coords.Coords(nodes[w1])
        d =c.distanceFromPoint(nodes[i]).reshape(-1)
        w2 = d < radii[w1] + radii[i]
        w = w1[w2]
        for j in w:
            connections = append(connections,i)
            connections = append(connections,j)
    connections = connections.reshape(-1,2)
    connections = removeDoubles(connections)
    return connections.reshape(-1,2)


def removeTriangles(elems):
    """Remove the triangles from the centerline.
    
    This is a clean-up function for the centerline.
    Triangles appearing in the centerline are removed by this function.
    Both input and output are the connectivity of the centerline.
    """
    rev = connectivity(elems).inverse()
    if rev.shape[1] > 2:
        w =  where(rev[:,-3] != -1)[0]
        for i in w:
            el = rev[i].compress(rev[i] != -1)
            u = unique(elems[el].reshape(-1))
            NB = u.compress(u != i)
            int = intersect1d(w,NB)
            if int.shape[0] == 2:
                tri = append(int,i)
                w1 = where(tri != tri.min())[0]
                t = (elems[:,0] == tri[w1[0]])*(elems[:,1] == tri[w1[1]])
                elems[t] = -1
    w2 = where(elems[:,0] != -1)[0]
    return elems[w2]
    

def centerline(fn):
    """Determine an approximated centerline corresponding with a triangulated surface.
    
    fn is the file name of a surface, including the extension (.off, .stl, .gts, .neu or .smesh)
    The output are the centerline nodes, an array containing the connectivity information
    and radii of the voronoi spheres.
    """
    nodesVor,radii= voronoiInner('%s' %fn)
    nodesC,radii=selectMaxVor(nodesVor,radii)
    elemsC = connectVorNodes(nodesC,radii)
    elemsC = removeTriangles(elemsC)
    return nodesC,elemsC,radii

# End

