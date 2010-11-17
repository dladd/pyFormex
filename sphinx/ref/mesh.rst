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
   :members: vectorRotation,sweepCoords,defaultEltype,mergeNodes,mergeMeshes,connectMesh,connectQuadraticMesh,connectMeshSequence

   ``Classes defined in module mesh``


   .. autoclass:: Mesh
      :members: setProp,getProp,maxProp,propSet,copy,toFormex,ndim,nelems,nplex,ncoords,shape,nedges,centroids,getCoords,getElems,getLowerEntitiesSelector,getLowerEntities,getEdges,getFaces,getAngles,getBorder,getBorderMesh,neighborsByNode,report,fuse,compact,select,unselect,meanNodes,addNodes,addMeanNodes,selectNodes,withProp,withoutProp,splitProp,convert,splitRandom,convertRandom,reduceDegenerate,splitDegenerate,renumber,extrude,revolve,sweep,test,clip,cclip,clipAtPlane,volumes,volume,equiAngleSkew,actor

   ``Functions defined in module mesh`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

