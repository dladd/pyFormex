.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse Mesh instances and mesh.mergeMeshes instead."'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse Mesh instances and mesh.mergeMeshes instead."'))))))))))))))))), (8, ')'), (4, '')))
.. $Id$  -*- rst -*-
.. pyformex reference manual --- fe
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-fe:

:mod:`fe` --- Finite Element Models in pyFormex.
================================================

.. automodule:: fe
   :synopsis: Finite Element Models in pyFormex.



   .. autoclass:: Model


      Model objects have the following methods:

      .. automethod:: nnodes()
      .. automethod:: nelems()
      .. automethod:: ngroups()
      .. automethod:: mplex()
      .. automethod:: splitElems(set)
      .. automethod:: elemNrs(group,set)
      .. automethod:: getElems(sets)
      .. automethod:: renumber(old=None,new=None)

**Functions defined in the module fe**

   .. autofunction:: mergeModels(femodels)
   .. autofunction:: mergedModel(meshes)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

