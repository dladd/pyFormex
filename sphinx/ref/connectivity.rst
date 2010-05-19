.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\n Use \'Connectivity.inverse()\' instead"'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\n Use \'Connectivity.inverse()\' instead"'))))))))))))))))), (8, ')'), (4, '')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'staticmethod')), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'staticmethod')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'staticmethod')), (4, '')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\n Use \'arraytools.inverseUniqueIndex()\' instead"'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\n Use \'arraytools.inverseUniqueIndex()\' instead"'))))))))))))))))), (8, ')'), (4, '')))
.. $Id$  -*- rst -*-
.. pyformex reference manual --- connectivity
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-connectivity:

:mod:`connectivity` --- A class and functions for handling nodal connectivity.
==============================================================================

.. automodule:: connectivity
   :synopsis: A class and functions for handling nodal connectivity.



   .. autoclass:: Connectivity


      Connectivity objects have the following methods:

      .. automethod:: nelems()
      .. automethod:: nplex()
      .. automethod:: encode(permutations=True,return_magic=False)
      .. automethod:: decode(codes,magic)
      .. automethod:: testDegenerate()
      .. automethod:: listDegenerate()
      .. automethod:: listNonDegenerate()
      .. automethod:: removeDegenerate()
      .. automethod:: testDoubles(permutations=True)
      .. automethod:: listUnique()
      .. automethod:: listDoubles()
      .. automethod:: removeDoubles(permutations=True)
      .. automethod:: selectNodes(nodsel)
      .. automethod:: insertLevel(nodsel)
      .. automethod:: untangle(ind=None)
      .. automethod:: tangle(lo)
      .. automethod:: inverse()

**Functions defined in the module connectivity**

   .. autofunction:: enmagic2(cols,magic=0)
   .. autofunction:: demagic2(codes,magic)
   .. autofunction:: enmagic(elems)
   .. autofunction:: demagic(codes,magic)
   .. autofunction:: inverseIndex(index,maxcon=4)
   .. autofunction:: reverseIndex(None)
   .. autofunction:: reverseUniqueIndex()
   .. autofunction:: adjacencyList(elems)
   .. autofunction:: adjacencyArray(elems,maxcon=5)
   .. autofunction:: adjacencyArrays(elems,nsteps=1)
   .. autofunction:: connected(index,i)
   .. autofunction:: adjacent(index,inv=None)
   .. autofunction:: closedLoop(elems)
   .. autofunction:: connectedLineElems(elems)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

