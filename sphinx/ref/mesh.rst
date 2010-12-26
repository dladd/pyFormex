DEBUG: Using the (slower) Python misc functions
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
   :members: vectorRotation,sweepCoords,defaultEltype,mergeNodes,mergeMeshes,connectMesh,connectQuadraticMesh

   ``Classes defined in module mesh``


   .. autoclass:: Mesh
      :members: affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,write,setProp,getProp,maxProp,propSet,copy,toFormex,nedges,centroids,getCoords,getElems,getLowerEntitiesSelector,getLowerEntities,getNodes,getPoints,getEdges,getFaces,getCells,getFaceEdges,getBorder,getBorderMesh,reverse,nodeConnections,nNodeConnected,nodeAdjacency,nNodeAdjacent,getAngles,node2nodeAdjacency,nNode2nodeAdjacent,avgNodalScalarOnAdjacentNodes,report,fuse,matchCoords,matchElemsCentroids,compact,select,cselect,meanNodes,addNodes,addMeanNodes,selectNodes,withProp,withoutProp,splitProp,splitRandom,convert,convertRandom,reduceDegenerate,splitDegenerate,renumber,renumberElems,extrude,revolve,sweep,concatenate,test,clip,cclip,clipAtPlane,areas,volumes,volume,equiAngleSkew,connect

   ``Functions defined in module mesh`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

