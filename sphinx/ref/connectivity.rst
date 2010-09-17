.. $Id$  -*- rst -*-
.. pyformex reference manual --- connectivity
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-connectivity:

:mod:`connectivity` --- A class and functions for handling nodal connectivity.
==============================================================================

.. automodule:: connectivity
   :synopsis: A class and functions for handling nodal connectivity.



   .. autoclass:: Connectivity
      :members: nelems,nplex,encode,testDegenerate,listDegenerate,listNonDegenerate,removeDegenerate,testDoubles,listUnique,listDoubles,removeDoubles,selectNodes,insertLevel,untangle,tangle,inverse

**Functions defined in the module connectivity**

   .. autofunction:: enmagic2(cols,magic=???)
   .. autofunction:: demagic2(codes,magic)
   .. autofunction:: enmagic(elems)
   .. autofunction:: demagic(codes,magic)
   .. autofunction:: inverseIndex(index,maxcon=???)
   .. autofunction:: adjacencyList(elems)
   .. autofunction:: adjacencyArray(elems,maxcon=???)
   .. autofunction:: adjacencyArrays(elems,nsteps=???)
   .. autofunction:: connected(index,i)
   .. autofunction:: adjacent(index,inv=???)
   .. autofunction:: closedLoop(elems)
   .. autofunction:: connectedLineElems(elems)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

