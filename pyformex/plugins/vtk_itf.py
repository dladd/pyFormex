# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Interface with VTK.

This module provides an interface with some function of the Python
Visualiztion Toolkit (VTK).
Documentation for VTK can be found on http://www.vtk.org/

This module provides the basic interface to convert data structures between
vtk and pyFormex.
"""

from vtk import *
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import create_vtk_array as cva
from vtk.util.numpy_support import get_numpy_array_type as gnat
from vtk.util.numpy_support import get_vtk_array_type as gvat
from numpy import *

from mesh import Mesh
from coords import Coords
from plugins.trisurface import TriSurface

import os

def cleanVPD(vpd):
    """Clean the vtkPolydata
    
    Clean the vtkPolydata, adjusting connectivity, removing duplicate elements
    and coords, renumbering the connectivity. This is often needed after
    setting the vtkPolydata, to make the vtkPolydata fit for use with other
    operations. Be aware that this operation will change the order and
    numbering of the original data.
    
    Parameters:

    - `vpd`: a vtkPolydata

    Returns the cleaned vtkPolydata.
    """
    cleaner=vtkCleanPolyData()
    cleaner.SetInput(vpd)
    cleaner.Update()
    return  cleaner.GetOutput()
    

def convert2VPD(M,clean=False):
    """Convert pyFormex data to vtkPolyData.
    
    Convert a pyFormex Mesh or Coords object into vtkPolyData.
    This is limited to vertices, lines, and polygons.
    Lines should already be ordered (with connectedLineElems for instance).

    Parameters:

    - `M`: a Mesh or Coords type. If M is a Coords type it will be saved as
      VERTS. Else...
    - `clean`: if True, the resulting vtkdata will be cleaned by calling
      cleanVPD.

    Returns a vtkPolyData.
    """
    
    print('STARTING CONVERSION FOR DATA OF TYPE %s '%type(M))
    
    if type(M) == Coords:
        M = Mesh(M,arange(M.ncoords()))
    
    Nelems = M.nelems() # Number of elements
    Ncxel = M.nplex() # # Number of nodes per element
    
    # create a vtkPolyData variable
    vpd=vtkPolyData()
    
    # creating  vtk coords
    pts = vtkPoints()
    ntype=gnat(pts.GetDataType())
    coordsv = n2v(asarray(M.coords,order='C',dtype=ntype),deep=1) #.copy() # deepcopy array conversion for C like array of vtk, it is necessary to avoid memry data loss
    pts.SetNumberOfPoints(M.ncoords())
    pts.SetData(coordsv)
    vpd.SetPoints(pts)
    
    
    # create vtk connectivity
    elms = vtkIdTypeArray()
    ntype=gnat(vtkIdTypeArray().GetDataType())
    elmsv = concatenate([Ncxel*ones(Nelems).reshape(-1,1),M.elems],axis=1)
    elmsv = n2v(asarray(elmsv,order='C',dtype=ntype),deep=1) #.copy() # deepcopy array conversion for C like array of vtk, it is necessary to avoid memry data loss
    elms.DeepCopy(elmsv)

    # set vtk Cell data
    datav = vtkCellArray()
    datav.SetCells(Nelems,elms)
    if Ncxel == 1:
        try:
            print("setting VERTS for data with %s maximum number of point for cell "%Ncxel)
            vpd.SetVerts(datav)
        except:
            raise ValueError,"Error in saving  VERTS"

    elif Ncxel == 2:
        try:
            print ("setting LINES for data with %s maximum number of point for cell "%Ncxel)
            vpd.SetLines(datav)
        except:
            raise  ValueError,"Error in saving  LINES"
            
    else:
        try:
            print ("setting POLYS for data with %s maximum number of point for cell "%Ncxel)
            vpd.SetPolys(datav)
        except:
            raise ValueError,"Error in saving  POLYS"
            
    vpd.Update()
    if clean:
        vpd=cleanVPD(vpd)
    return vpd
    

def convertVPD2Triangles(vpd):
    """Convert a vtkPolyData to a vtk triangular surface.
    
    Convert a vtkPolyData to a vtk triangular surface. This is convenient
    when vtkPolyData are non-triangular polygons.

    Parameters:

    - `vpd`: a vtkPolyData

    Returns
    """
    triangles = vtkTriangleFilter()
    triangles.SetInput(vpd)
    triangles.Update()
    return triangles.GetOutput()


def convertFromVPD(vpd):
    """Convert a vtkPolyData into pyFormex objects.
    
    Convert a vtkPolyData into pyFormex objects.

    Parameters:

    - `vpd`: a vtkPolyData

    Returns a tuple with points, polygons, lines, vertices numpy arrays.
    Returns None for the missing data.
    """
    pts=polys=lines=verts=None

    # getting points coords
    if  vpd.GetPoints().GetData().GetNumberOfTuples():
        ntype=gnat(vpd.GetPoints().GetDataType())
        pts = asarray(v2n(vpd.GetPoints().GetData()),dtype=ntype)
        print('Saved points coordinates array')
        
    # getting Polygons
    if  vpd.GetPolys().GetData().GetNumberOfTuples():
        ntype=gnat(vpd.GetPolys().GetData().GetDataType())
        Nplex = vpd.GetPolys().GetMaxCellSize()
        polys = asarray(v2n(vpd.GetPolys().GetData()),dtype=ntype).reshape(-1,Nplex+1)[:,1:]
        print('Saved polys connectivity array')
        
    # getting Lines
    if  vpd.GetLines().GetData().GetNumberOfTuples():
        ntype=gnat(vpd.GetLines().GetDataType())
        Nplex = vpd.GetLines().GetMaxCellSize()
        lines = asarray(v2n(vpd.GetLines().GetData()),dtype=ntype).reshape(-1,Nplex+1)[:,1:]
        print('Saved lines connectivity array')
        
    # getting Vertices
    if  vpd.GetVerts().GetData().GetNumberOfTuples():
        ntype=gnat(vpd.GetVerts().GetDataType())
        Nplex = vpd.GetVerts().GetMaxCellSize()
        verts = asarray(v2n(vpd.GetVerts().GetData()),dtype=ntype).reshape(-1,Nplex+1)[:,1:]
        print('Saved verts connectivity array')
        
    return pts, polys, lines, verts

# End
