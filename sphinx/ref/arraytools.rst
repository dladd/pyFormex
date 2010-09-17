.. $Id$  -*- rst -*-
.. pyformex reference manual --- arraytools
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-arraytools:

:mod:`arraytools` --- A collection of numerical array utilities.
================================================================

.. automodule:: arraytools
   :synopsis: A collection of numerical array utilities.


   .. autofunction:: niceLogSize(f)
   .. autofunction:: niceNumber(f,approx=???)
   .. autofunction:: sind(arg,angle_spec=???)
   .. autofunction:: cosd(arg,angle_spec=???)
   .. autofunction:: tand(arg,angle_spec=???)
   .. autofunction:: dotpr(A,B,axis=???)
   .. autofunction:: length(A,axis=???)
   .. autofunction:: normalize(A,axis=???)
   .. autofunction:: projection(A,B,axis=???)
   .. autofunction:: norm(v,n=???)
   .. autofunction:: solveMany(A,b)
   .. autofunction:: permutations(iterable,r=???)
   .. autofunction:: inside(p,mi,ma)
   .. autofunction:: isClose(values,target,rtol=???,atol=???)
   .. autofunction:: unitVector(v)
   .. autofunction:: rotationMatrix(angle,axis=???,angle_spec=???)
   .. autofunction:: rotMatrix(u,w=???,n=???)
   .. autofunction:: rotationAnglesFromMatrix(mat,angle_spec=???)
   .. autofunction:: growAxis(a,add,axis=???,fill=???)
   .. autofunction:: reverseAxis(a,axis=???)
   .. autofunction:: addAxis(a,axis=???)
   .. autofunction:: stack(al,axis=???)
   .. autofunction:: checkArray(a,shape=???,kind=???,allow=???)
   .. autofunction:: checkArray1D(a,size=???,kind=???,allow=???)
   .. autofunction:: checkUniqueNumbers(nrs,nmin=???,nmax=???)
   .. autofunction:: readArray(file,dtype,shape,sep=???)
   .. autofunction:: writeArray(file,array,sep=???)
   .. autofunction:: cubicEquation(a,b,c,d)
   .. autofunction:: unique1dOrdered(ar1,return_index=???,return_inverse=???)
   .. autofunction:: renumberIndex(index)
   .. autofunction:: inverseUniqueIndex(index)
   .. autofunction:: sortByColumns(A)
   .. autofunction:: uniqueRows(A)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

