.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse mesh.connectMesh instead."'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse mesh.connectMesh instead."'))))))))))))))))), (8, ')'), (4, '')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse mesh.sweepMesh instead."'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse mesh.sweepMesh instead."'))))))))))))))))), (8, ')'), (4, '')))
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

      .. automethod:: copy()
      .. automethod:: toFormex()
      .. automethod:: data()
      .. automethod:: nelems()
      .. automethod:: nplex()
      .. automethod:: ncoords()
      .. automethod:: shape()
      .. automethod:: bbox()
      .. automethod:: center()
      .. automethod:: nedges()
      .. automethod:: centroids()
      .. automethod:: report()
      .. automethod:: fuse()
      .. automethod:: compact()
      .. automethod:: select(selected,compact=False)
      .. automethod:: meanNodes(nodsel)
      .. automethod:: addNodes(newcoords,eltype=None)
      .. automethod:: addMeanNodes(nodsel,eltype=None)
      .. automethod:: selectNodes(nodsel,eltype)
      .. automethod:: convert(totype)
      .. automethod:: randomSplit(n)
      .. automethod:: convertRandom(choices)
      .. automethod:: extrude(n,step=1.,dir=0,autofix=True)
      .. automethod:: sweep(path,autofix=True)
      .. automethod:: concatenate(clas,meshes)

**Functions defined in the module mesh**

   .. autofunction:: vectorRotation(vec1,vec2,upvec=[0.,0.,1.])
   .. autofunction:: sweepCoords(path,origin=[0.,0.,0.],normal=0,upvector=2,avgdir=False,enddir=None)
   .. autofunction:: defaultEltype(nplex)
   .. autofunction:: mergeNodes(nodes)
   .. autofunction:: mergeMeshes(meshes)
   .. autofunction:: connectMesh(mesh1,mesh2,n=1,n1=None,n2=None,eltype=None)
   .. autofunction:: connectMeshSequence(ML,loop=False)
   .. autofunction:: createWedgeElements(S1,S2,div=1)
   .. autofunction:: sweepGrid(nodes,elems,path,scale=1.,angle=0.,a1=None,a2=None)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

