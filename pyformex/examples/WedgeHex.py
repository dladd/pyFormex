#!/usr/bin/env pyformex --gui
# $Id: Isopar.py 921 2009-02-24 09:51:50Z bverheg $
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""WedgeHex

level = 'advanced'
topics = ['FEA']
techniques = ['meshing'] 
"""

import simple
import elements
from plugins.mesh import *
from plugins.fe import *

# there are 3 basic functions: 
#A) generate a mesh by revolving a 2D mesh (currently it support only Quad, but can be easily extend to Tri)
#B)check a HEX mesh and if there are degenereted HEX transform them into WEDGES
#C)check a HEX grid: if the HEX has a negative volume re-numbers its node to get a positive volume (negative volume appears 1-by reflecting, 2-by revolving a mesh by a negative angle, 3-by extruding or sweeping a mesh backwards).



def Hex2Wedge(v):
    """it takes the vertices of an Hex and if it detects a null area it transform the Hex into a Wedge"""
    f=array(elements.Hex8().faces)
    def areaTri(x):#from surface plugin
       return 0.5*vectorPairAreaNormals([x[1]-x[0]],[x[2]-x[1]])[0]
    def areaQuad(x):
        return areaTri(x[[0, 1, 2]])+areaTri(x[[0, 2,3]])
    
    degfac=None
    for i in range(f.shape[0]):
        if areaQuad(v[f[i]])==0:
            #print('Hex with degenerated face %d'%i)
            degfac=i
    if degfac==None:return v#it is a real Hex8

    de= f[degfac]#degen vert
    
    p0=v[de[0]]
    for i in range(3):
        if all(v[de][i+1]==p0)==False:pde=[0, i]
    
    v0=v[de[pde][0]]
    v1=v[de[pde][1]]
    
    SideQuad=[]
    for indf in range(6):
        inside0, inside1=False, False
        for iv in range(4):
           if  all(v[f[indf][iv]]==v0)==True:
               inside0=True
        for iv in range(4):
            if  all(v[f[indf][iv]]==v1)==True:
                inside1=True
        if (inside0, inside1)==(True, True):SideQuad.append(indf)
    
    if degfac%2==0:opposfac=degfac+1
    else: opposfac=degfac-1
    faceTri=[]
    for i in range(6):
        if i not in SideQuad: 
            if i!=opposfac: faceTri.append(i)
    w0, w1=[], []
    for i in range(4):
        if f[faceTri[0]][i]!=de[pde][0]:
            if f[faceTri[0]][i]!=de[pde][1]:w0.append(f[faceTri[0]][i])
            
        if f[faceTri[1]][i]!=de[pde][0]:
            if f[faceTri[1]][i]!=de[pde][1]:w1.append(f[faceTri[1]][i])
    
    
    
    l0=array([ length(v[w1[i]]-v[w0[0]]) for i in range(3)])
    l1=array([ length(v[w1[i]]-v[w0[1]]) for i in range(3)])
    l2=array([ length(v[w1[i]]-v[w0[2]]) for i in range(3)])
    w1t=array([where(l0==l0.min())[0]])
    w1t=append(w1t, where(l1==l1.min())[0])
    w1t=append(w1t, where(l2==l2.min())[0])
    
    w1=array(w1)[w1t]
    w= append(w0 , w1)
    w=v[w]
    clear()
    W=Formex([w], eltype='Wedge6')
    #drawNumbers(W.points())
    #draw(W)
    W=correctWedgeDirection(W[:].reshape(-1, 3))[0]

    return W
    #W=Formex([W], eltype='Wedge6')
    #return W#it is not an Hex8, but a Wedge 6



def triplescalarproduct(u, v, w):
    n= cross(u, v)
    return dot(n,w )
def correctHexDirection(h):
    """it takes a -1,3 array with the 8 points of an hexahedron as defined in pyformex and, if the convention of pyformex does not match the convention of Abaqus, reorder the vertices for Abaqus. False is returned if the elements has been corrected, False if it zas already correct"""
    tsp=triplescalarproduct(h[1]-h[0], h[2]-h[1], h[4]-h[0])
    if tsp>0:
        return h, False
    if tsp<0:
        return h[[3, 2, 1, 0, 7, 6, 5, 4]], True
        
def correctHexMeshDirection(M, drawit=False):
    """it takes a -1,8,3 Formex (Hex mesh) from pyformex and, if the convention of pyformex does not match the convention of Abaqus, reorder the vertices for Abaqus. Returns the corresponding Hex mesh. It also return the list of the modified element. If drawit==True it also draws red points around the modifiend elements."""
    m=M[:].copy()
    cor_el_list=zeros([M[:].shape[0]])
    for i in range(m.shape[0]):
        m[i], modified=correctHexDirection(m[i])
        if modified==True:cor_el_list[i]=1.
    modified_indices=where(cor_el_list==1)[0]
    if drawit==True: draw(Formex(M[:][modified_indices].reshape(-1, 1, 3)), marksize=10, color='red')
    print('correcting HEX :%d HEX have been corrected'%modified_indices.shape[0])
    return Formex(m.reshape(-1, 8, 3), eltype='Hex8')



def correctWedgeDirection(h):
    """it takes a -1,3 array with the 8 points of an hexahedron as defined in pyformex and, if the convention of pyformex does not match the convention of Abaqus, reorder the vertices for Abaqus. False is returned if the elements has been corrected, False if it zas already correct"""
    tsp=triplescalarproduct(h[1]-h[0], h[2]-h[1], h[5]-h[0])
    if tsp>0:
        return h, False
    if tsp<0:
        return h[[3, 4, 5, 0, 1,2]], True
        
def correctWedgeMeshDirection(M, drawit=False):
    """it takes a -1,6,3 Formex (Hex mesh) from pyformex and, if the convention of pyformex does not match the convention of Abaqus, reorder the vertices for Abaqus. Returns the corresponding Wedge mesh. It also return the list of the modified element. If drawit==True it also draws red points around the modifiend elements."""
    m=M[:].copy()
    cor_el_list=zeros([M[:].shape[0]])
    for i in range(m.shape[0]):
        m[i], modified=correctWedgeDirection(m[i])
        if modified==True:cor_el_list[i]=1.
    modified_indices=where(cor_el_list==1)[0]
    if drawit==True: draw(Formex(M[:][modified_indices].reshape(-1, 1, 3)), marksize=10, color='red')
    #print('the number of uncorrect Wedges was %d'%modified_indices.shape)
    return Formex(m.reshape(-1, 6, 3), eltype='Wedge6')

def detectHex2Wedge(W):
    w=W[:].copy()
    we=[]
    he=[]
    for i in range(w.shape[0]):
        current_el=Hex2Wedge(w[i])[:]
        if current_el.shape[0]==6:we.append(current_el)#it is wedge6
        if current_el.shape[0]==8:he.append(current_el)#it is wedge6
    we, he=array(we), array(he)
    print('detecting degenerated HEX: there are %d hex and %d wedge'%(he.shape[0], we.shape[0]))
    return array(we), array(he)
def revolve_QuadMesh(n0, e0, nr, ang=None):
    """it takes a Quad mesh on xy and revolve it along the z axis nr times by ang. If ang==None, then it is calculated in order to fill 360 degrees with the nr revolutions."""
    if ang==None: ang=360./nr
    for i in range(int(nr)):
        n1=Formex(n0).rotate(-ang, 1)[:].reshape(-1, 3)
        n, e=connectMesh(n0,n1,e0)
        n0=n1.copy()
        parts.append(Formex(n[e], eltype='Hex8'))
    femodels = [part.feModel() for part in parts]
    nodes,elems = mergeModels(femodels)
    elems=concatenate([elems], 0).reshape(-1, 8)
    return nodes, elems
clear()
#create a 2D xy mesh
n=4
G=simple.rectangle(1,1,1.,1.).replic(n,1.,dir=1).replic(n,1.,dir=0)
draw(G, color='red')
view('front')
sleep(1)
#create a 3D axial-symmetric mesh by REVOLVING
n0, e0=G.feModel()
parts=[]



n, e=revolve_QuadMesh(n0, e0, nr=4, ang=20)
C=Formex(n[e], eltype='Hex8')

#check if there are Wedge elements in the global Hex mesh
w, h= detectHex2Wedge(C)
W=Formex(w, eltype='Wedge6')
H=Formex(h, eltype='Hex8')

view('iso')
WE=draw(W, color='blue')
sleep(2)
undraw(WE)
draw(H, color='red')
sleep(2)
draw(W, color='blue')
draw(H, color='red')

sleep(2)

clear()
#demonstration of negative HEX volumes : need to correct their orientation
Hr=H.reflect()
H2=H+Hr
H=correctHexMeshDirection(H2, True)
draw(H2)
sleep(2)
clear()
n, e=revolve_QuadMesh(n0, e0, nr=4, ang=-20)#negative angle
Crev=Formex(n[e], eltype='Hex8')
w, hrev= detectHex2Wedge(Crev)
Hrev=Formex(hrev, eltype='Hex8')
Hok=correctHexMeshDirection(Hrev, True)
draw(Hok, color='gren')
