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
      .. automethod:: Max()
      .. automethod:: unique()
      .. automethod:: checkUnique()
      .. automethod:: check()
      .. automethod:: reverseIndex()
      .. automethod:: expand()

   Functions defined in the connectivity module:

   .. autofunction:: magic_numbers(elems,magic)
   .. autofunction:: demagic(mag,magic)
   .. autofunction:: expandElems(elems)
   .. autofunction:: compactElems(edges,faces)
   .. autofunction:: reverseUniqueIndex(index)
   .. autofunction:: reverseIndex(index,maxcon=3)
   .. autofunction:: adjacencyList(elems)
   .. autofunction:: adjacencyArray(elems,maxcon=3,neighbours=1)
   .. autofunction:: connected(index,i)
   .. autofunction:: adjacent(index,rev=None)
   .. autofunction:: closedLoop(elems)
   .. autofunction:: connectedLineElems(elems)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

