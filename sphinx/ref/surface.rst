.. $Id$  -*- rst -*-
.. pyformex reference manual --- surface
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-surface:

:mod:`surface` --- Operations on triangulated surfaces.
=======================================================

.. automodule:: surface
   :synopsis: Operations on triangulated surfaces.



   .. autoclass:: TriSurface


      TriSurface objects have the following methods:

      .. automethod:: nedges()
      .. automethod:: nfaces()
      .. automethod:: vertices()
      .. automethod:: shape()
      .. automethod:: getEdges()
      .. automethod:: getFaces()
      .. automethod:: setCoords(coords)
      .. automethod:: setElems(elems)
      .. automethod:: setEdgesAndFaces(edges,faces)
      .. automethod:: refresh()
      .. automethod:: append(S)
      .. automethod:: copy()
      .. automethod:: select(idx,compact=True)
      .. automethod:: pointNormals()
      .. automethod:: offset(distance=1.)
      .. automethod:: read(clas,fn,ftype=None)
      .. automethod:: write(fname,ftype=None)
      .. automethod:: reflect(None)
      .. automethod:: avgVertexNormals()
      .. automethod:: areaNormals()
      .. automethod:: facetArea()
      .. automethod:: area()
      .. automethod:: volume()
      .. automethod:: curvature(neighbours=1)
      .. automethod:: inertia()
      .. automethod:: edgeConnections()
      .. automethod:: nodeConnections()
      .. automethod:: nEdgeConnected()
      .. automethod:: nNodeConnected()
      .. automethod:: edgeAdjacency()
      .. automethod:: nEdgeAdjacent()
      .. automethod:: nodeAdjacency()
      .. automethod:: nNodeAdjacent()
      .. automethod:: surfaceType()
      .. automethod:: borderEdges()
      .. automethod:: borderEdgeNrs()
      .. automethod:: borderNodeNrs()
      .. automethod:: isManifold()
      .. automethod:: isClosedManifold()
      .. automethod:: checkBorder()
      .. automethod:: fillBorder(method=0)
      .. automethod:: border()
      .. automethod:: edgeCosAngles()
      .. automethod:: edgeAngles()
      .. automethod:: aspectRatio()
      .. automethod:: smallestAltitude()
      .. automethod:: longestEdge()
      .. automethod:: shortestEdge()
      .. automethod:: stats()
      .. automethod:: distanceOfPoints(X,return_points=False)
      .. automethod:: edgeFront(startat=0,okedges=None,front_increment=1)
      .. automethod:: nodeFront(startat=0,front_increment=1)
      .. automethod:: walkEdgeFront(startat=0,nsteps=1,okedges=None,front_increment=1)
      .. automethod:: walkNodeFront(startat=0,nsteps=1,front_increment=1)
      .. automethod:: growSelection(sel,mode='node',nsteps=1)
      .. automethod:: partitionByEdgeFront(okedges,firstprop=0,startat=0)
      .. automethod:: partitionByNodeFront(firstprop=0,startat=0)
      .. automethod:: partitionByConnection()
      .. automethod:: partitionByAngle(angle=180.,firstprop=0,startat=0)
      .. automethod:: cutWithPlane(None)
      .. automethod:: connectedElements(target,elemlist=None)
      .. automethod:: intersectionWithPlane(p,n,atol=0.,ignoreErrors=False)
      .. automethod:: slice(dir=0,nplanes=20,ignoreErrors=False)
      .. automethod:: smoothLowPass(n_iterations=2,lambda_value=0.5,neighbours=1)
      .. automethod:: smoothLaplaceHC(n_iterations=2,lambda_value=0.5,alpha=0.,beta=0.2,neighbours=1)
      .. automethod:: check(verbose=False)
      .. automethod:: split(base,verbose=False)
      .. automethod:: coarsen(min_edges=None,max_cost=None,mid_vertex=False,length_cost=False,max_fold=1.0,volume_weight=0.5,boundary_weight=0.5,shape_weight=0.0,progressive=False,log=False,verbose=False)
      .. automethod:: refine(max_edges=None,min_cost=None,log=False,verbose=False)
      .. automethod:: smooth(lambda_value=0.5,n_iterations=2,fold_smoothing=None,verbose=False)
      .. automethod:: boolean(surf,op,inter=False,check=False,verbose=False)

**Functions defined in the module surface**

   .. autofunction:: areaNormals(x)
   .. autofunction:: stlConvert(stlname,outname=None,options='-d')
   .. autofunction:: read_gts(fn)
   .. autofunction:: read_off(fn)
   .. autofunction:: read_stl(fn,intermediate=None)
   .. autofunction:: read_gambit_neutral(fn)
   .. autofunction:: write_gts(fn,nodes,edges,faces)
   .. autofunction:: write_stla(f,x)
   .. autofunction:: write_stlb(f,x)
   .. autofunction:: write_gambit_neutral(fn,nodes,elems)
   .. autofunction:: write_off(fn,nodes,elems)
   .. autofunction:: write_smesh(fn,nodes,elems)
   .. autofunction:: surface_volume(x,pt=None)
   .. autofunction:: curvature(coords,elems,edges,neighbours=1)
   .. autofunction:: surfaceInsideLoop(coords,elems)
   .. autofunction:: fillHole(coords,elems)
   .. autofunction:: create_border_triangle(coords,elems)
   .. autofunction:: read_error(cnt,line)
   .. autofunction:: degenerate(area,norm)
   .. autofunction:: read_stla(fn,dtype=Float,large=False,guess=True)
   .. autofunction:: read_ascii_large(fn,dtype=Float)
   .. autofunction:: off_to_tet(fn)
   .. autofunction:: find_row(mat,row,nmatch=None)
   .. autofunction:: find_nodes(nodes,coords)
   .. autofunction:: find_first_nodes(nodes,coords)
   .. autofunction:: find_triangles(elems,triangles)
   .. autofunction:: remove_triangles(elems,remove)
   .. autofunction:: Rectangle(nx,ny)
   .. autofunction:: Cube()
   .. autofunction:: Sphere(level=4,verbose=False,filename=None)
   .. autofunction:: checkDistanceLinesPointsTreshold(p,q,m,dtresh)
   .. autofunction:: intersectLineWithPlaneOne2One(q,m,p,n)
   .. autofunction:: checkPointInsideTriangleOne2One(tpi,pi,atol=1.e-5)
   .. autofunction:: intersectSurfaceWithLines(ts,qli,mli)
   .. autofunction:: intersectSurfaceWithSegments(s1,segm,atol=1.e-5)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

