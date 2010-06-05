.. $Id$  -*- rst -*-
.. pyformex reference manual --- mesh
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-mesh:

:mod:`mesh` --- mesh.py
=======================

.. automodule:: mesh
   :synopsis: mesh.py



   .. autoclass:: Mesh


      Mesh objects have the following methods:

      .. automethod:: setCoords(coords)
      .. automethod:: setProp(prop=None)
      .. automethod:: getProp()
      .. automethod:: maxProp()
      .. automethod:: propSet()
      .. automethod:: copy()
      .. automethod:: toFormex()
      .. automethod:: ndim()
      .. automethod:: nelems()
      .. automethod:: nplex()
      .. automethod:: ncoords()
      .. automethod:: shape()
      .. automethod:: nedges()
      .. automethod:: centroids()
      .. automethod:: getCoords()
      .. automethod:: getElems()
      .. automethod:: getLowerEntitiesSelector(level=1,unique=False)
      .. automethod:: getLowerEntities(level=1,unique=False)
      .. automethod:: getEdges(unique=False)
      .. automethod:: getFaces(unique=False)
      .. automethod:: getAngles(angle_spec=Deg)
      .. automethod:: getBorder()
      .. automethod:: getBorderMesh()
      .. automethod:: report()
      .. automethod:: fuse()
      .. automethod:: compact()
      .. automethod:: select(selected)
      .. automethod:: meanNodes(nodsel)
      .. automethod:: addNodes(newcoords,eltype=None)
      .. automethod:: addMeanNodes(nodsel,eltype=None)
      .. automethod:: selectNodes(nodsel,eltype)
      .. automethod:: withProp(val)
      .. automethod:: splitProp()
      .. automethod:: convert(totype)
      .. automethod:: splitRandom(n)
      .. automethod:: convertRandom(choices)
      .. automethod:: reduceDegenerate(eltype=None)
      .. automethod:: splitDegenerate(autofix=True)
      .. automethod:: renumber(order='elems')
      .. automethod:: extrude(n,step=1.,dir=0,autofix=True)
      .. automethod:: revolve(n,axis=0,angle=360.,around=None,autofix=True)
      .. automethod:: sweep(path,autofix=True)
      .. automethod:: concatenate(clas,meshes)
      .. automethod:: test(nodes='all',dir=0,min=None,max=None,atol=0.)
      .. automethod:: clip(t)
      .. automethod:: cclip(t)
      .. automethod:: clipAtPlane(p,n,nodes='any',side='+')
      .. automethod:: equiAngleSkew()

**Functions defined in the module mesh**

   .. autofunction:: vectorRotation(vec1,vec2,upvec=[0.,0.,1.])
   .. autofunction:: sweepCoords(path,origin=[0.,0.,0.],normal=0,upvector=2,avgdir=False,enddir=None,scalex=None,scaley=None)
   .. autofunction:: defaultEltype(nplex)
   .. autofunction:: mergeNodes(nodes)
   .. autofunction:: mergeMeshes(meshes)
   .. autofunction:: connectMesh(mesh1,mesh2,n=1,n1=None,n2=None,eltype=None)
   .. autofunction:: connectMeshSequence(ML,loop=False)
   .. autofunction:: structuredHexGrid(dx,dy,dz,isophex='hex64')
   .. autofunction:: correctHexMeshOrientation(hm)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

