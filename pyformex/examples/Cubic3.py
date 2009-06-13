#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""Cubic3

level = 'normal'
topics = ['geometry','surface']
techniques = []

"""

from numpy import *

def p4cubic (A, B, C, D, derivate=False):
    "if derivate==False: it takes 4 points and returns the coefficient (a,b,c,d) of the cubic curve passing through them. y=ax**3+bx**2+cx+d. If derivate == True it returns also the dervative at point B"
    coefficients=array([[A[0]**3, A[0]**2, A[0], 1], [B[0]**3, B[0]**2, B[0], 1], [C[0]**3, C[0]**2, C[0], 1], [D[0]**3, D[0]**2, D[0], 1]])
    data=array([A[1], B[1], C[1], D[1]])
    a, b, c, d=linalg.solve(coefficients, data)    
    if derivate==False: return array([a, b, c, d])
    if derivate==True: return 3*a*B[0]**2 + 2*b*B[0]+c

def d3pcubic(derA, A, B, C, derivate=False):
    "if derivate ==False it gives the coefficient (a,b,c,d) of the cubic curve passing through point A,B,C with a derivative in A equal to der. y=ax**3+bx**2+cx+d. If derivate ==True it also gives the derivative at the second point (B)"
    coefficients=array([[3*A[0]**2, 2*A[0], 1, 0], [A[0]**3, A[0]**2, A[0], 1], [B[0]**3, B[0]**2, B[0], 1], [C[0]**3, C[0]**2, C[0], 1]])
    data=array([derA, A[1], B[1], C[1]])    
    a, b, c, d=linalg.solve(coefficients, data)
    if derivate==False: return array([a, b, c, d])
    if derivate==True: return 3*a*B[0]**2 + 2*b*B[0]+c
    
def cubicSpline (nodes):
    "takes an array of n-nodes (0...n-1) and returns the cofficients of 1+(n-3)/2 cubic curves. The first one pass through the first 4 points. The other curves pass through last point of the previous and two new points."
    curves=array([p4cubic(nodes[0], nodes[1], nodes[2], nodes[3])]) #deve essere un array 2D. ogni elemento e' fatto da 4 numeri che sono i coefficienti du una curva cubica. quindi ci sono tente cubiche a pezzi
    deriv=p4cubic(nodes[0], nodes[1], nodes[2], nodes[3], True)
    for i in range(1, nodes.shape[0]-2, 1):
        curves=append(curves, array([d3pcubic(deriv, nodes[i], nodes[i+1], nodes[i+2])]), axis=0)
        deriv=d3pcubic(deriv, nodes[i], nodes[i+1], nodes[i+2], derivate=True)
    return curves
    
def createOrdinate (x, coef):    
    return coef[0]*x**3+coef[1]*x**2+coef[2]*x+coef[3]
    
def interpolate2D(nodes,n=5):
    "n=nuber of points between 2 consecutive points. IMPORTANT: nodes have to be 2D: no z coordinates"
    if nodes.shape[1]==3: raise RuntimeError, "Nodes has to be (-1,2) array with x crescente"
    plist=array([])
    curv=cubicSpline(nodes)       
    #after the first 4 points: two points belong to the same curve
    for index in range(0, nodes.shape[0]-2, 1):#e' l indice del punto che uso anche per vedere a quale curva corrisponde     
        for j in range(0, n):
            absci=nodes[index][0]+j*(nodes[index+1][0]-nodes[index][0])/n
            plist=append(plist,  absci)
            plist=append(plist,  createOrdinate(absci, curv[index]))            
    for j in range(0, n+1):
        absci=nodes[-2][0]+j*(nodes[-1][0]-nodes[-2][0])/n        
        plist=append(plist,  absci)
        plist=append(plist,  createOrdinate(absci, curv[-1]))          
    return plist.reshape(-1, 2)
   
def interpolate3D(nodes, n=5):
    "it takes an array of nodes and create a cubic interpolation through them. Between each couple it created n intervals. The x of the nodes has to be monotone."
    nodxy=delete(nodes, 2, axis=1) #extract x and y (beacause the interpolate works in 2D)
    newnodesy=interpolate2D(nodxy, n)       
    nodxz=delete(nodes, 1, axis=1) 
    newnodesz=interpolate2D(nodxz, n)
    return column_stack((newnodesz[:, 0], newnodesy[:, 1], newnodesz[:, 1])) 

nod3D=array([[0., 1, 1], [1, 0.1, 2], [2, -1, 1], [3, 0.3, -.5], [4, 0, 0], [5, -2, -2], [6, 3, -3]])
OLD=Formex(nod3D.reshape(-1, 1, 3))
draw (OLD, color='red')
drawNumbers (OLD)

newpoints= interpolate3D(nod3D, 3)
NEW=Formex(newpoints.reshape(1, -1, 3))
draw(NEW)

# End
