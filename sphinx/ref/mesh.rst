.. $Id$  -*- rst -*-
.. pyformex reference manual --- mesh
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-mesh:

:mod:`mesh` --- Finite element meshes in pyFormex.
==================================================

.. automodule:: mesh
   :synopsis: Finite element meshes in pyFormex.
   :members: vectorRotation,sweepCoords,defaultEltype,mergeNodes,mergeMeshes,connectMesh,connectQuadraticMesh,connectMeshSequence

   ``Classes defined in module mesh``


   .. autoclass:: Mesh
      :members: setProp,getProp,maxProp,propSet,copy,toFormex,ndim,nelems,nplex,ncoords,shape,nedges,centroids,getCoords,getElems,getLowerEntitiesSelector,getLowerEntities,getNodes,getPoints,getEdges,getFaces,getCells,getFaceEdges,getBorder,getBorderMesh,reverse,getAngles,neighborsByNode,report,fuse,compact,select,cselect,meanNodes,addNodes,addMeanNodes,selectNodes,withProp,withoutProp,splitProp,splitRandom,convert,convertRandom,reduceDegenerate,splitDegenerate,renumber,extrude,revolve,sweep,test,clip,cclip,clipAtPlane,areas,volumes,volume,equiAngleSkew,actor

   ``Functions defined in module mesh`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

