.. $Id$  -*- rst -*-
.. pyformex reference manual --- mesh
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-mesh:

:mod:`mesh` --- mesh.py
=======================

.. automodule:: mesh
   :synopsis: mesh.py



   .. autoclass:: Mesh
      :members: setCoords,setProp,getProp,maxProp,propSet,copy,toFormex,ndim,nelems,nplex,ncoords,shape,nedges,centroids,getCoords,getElems,getLowerEntitiesSelector,getLowerEntities,getEdges,getFaces,getAngles,getBorder,getBorderMesh,report,fuse,compact,select,unselect,meanNodes,addNodes,addMeanNodes,selectNodes,withProp,splitProp,convert,splitRandom,convertRandom,reduceDegenerate,splitDegenerate,renumber,extrude,revolve,sweep,test,clip,cclip,clipAtPlane,volumes,volume,equiAngleSkew,actor

**Functions defined in the module mesh**

   .. autofunction:: vectorRotation(vec1,vec2,upvec=???)
   .. autofunction:: sweepCoords(path,origin=???,normal=???,upvector=???,avgdir=???,enddir=???,scalex=???,scaley=???)
   .. autofunction:: defaultEltype(nplex)
   .. autofunction:: mergeNodes(nodes)
   .. autofunction:: mergeMeshes(meshes)
   .. autofunction:: connectMesh(mesh1,mesh2,n=???,n1=???,n2=???,eltype=???)
   .. autofunction:: connectMeshSequence(ML,loop=???)
   .. autofunction:: structuredHexGrid(dx,dy,dz,isophex=???)
   .. autofunction:: correctHexMeshOrientation(hm)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

