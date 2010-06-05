.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse cutWithPlane() instead. Check arguments "'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"\\nUse cutWithPlane() instead. Check arguments "'))))))))))))))))), (8, ')'), (4, '')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'functionWasRenamed')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (1, 'reverse'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'functionWasRenamed')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'functionWasRenamed')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (1, 'reverse'))))))))))))))))), (8, ')'), (4, '')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"feModel() is deprecated. Use toMesh() wherever possible.\\nfuse() remains available with the same result as feModel()"'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"feModel() is deprecated. Use toMesh() wherever possible.\\nfuse() remains available with the same result as feModel()"'))))))))))))))))), (8, ')'), (4, '')))
.. $Id$  -*- rst -*-
.. pyformex reference manual --- formex
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-formex:

:mod:`formex` --- Formex algebra in Python
==========================================

.. automodule:: formex
   :synopsis: Formex algebra in Python



   .. autoclass:: Formex


      Formex objects have the following methods:

      .. automethod:: setCoords(coords)
      .. automethod:: element(i)
      .. automethod:: point(i,j)
      .. automethod:: coord(i,j,k)
      .. automethod:: nelems()
      .. automethod:: nplex()
      .. automethod:: ndim()
      .. automethod:: npoints()
      .. automethod:: shape()
      .. automethod:: view()
      .. automethod:: getProp(index=None)
      .. automethod:: maxProp()
      .. automethod:: propSet()
      .. automethod:: centroids()
      .. automethod:: fuse(repeat=True,nodesperbox=1,rtol=1.e-5,atol=None)
      .. automethod:: toMesh(None)
      .. automethod:: info()
      .. automethod:: point2str(clas,point)
      .. automethod:: element2str(clas,elem)
      .. automethod:: asFormex()
      .. automethod:: asFormexWithProp()
      .. automethod:: asArray()
      .. automethod:: setPrintFunction(clas,func)
      .. automethod:: fprint(None)
      .. automethod:: setProp(p=None)
      .. automethod:: append(F)
      .. automethod:: concatenate(clas,Flist)
      .. automethod:: select(idx)
      .. automethod:: selectNodes(idx)
      .. automethod:: points()
      .. automethod:: vertices()
      .. automethod:: remove(F)
      .. automethod:: whereProp(val)
      .. automethod:: withProp(val)
      .. automethod:: splitProp()
      .. automethod:: elbbox()
      .. automethod:: unique(rtol=1.e-4,atol=1.e-6)
      .. automethod:: reverse()
      .. automethod:: test(nodes='all',dir=0,min=None,max=None,atol=0.)
      .. automethod:: clip(t)
      .. automethod:: cclip(t)
      .. automethod:: mirror(dir=2,pos=0,keep_orig=True)
      .. automethod:: centered()
      .. automethod:: resized(size=1.,tol=1.e-5)
      .. automethod:: circulize(angle)
      .. automethod:: circulize1()
      .. automethod:: shrink(factor)
      .. automethod:: replic(n,step,dir=0)
      .. automethod:: replic2(n1,n2,t1=1.0,t2=1.0,d1=0,d2=1,bias=0,taper=0)
      .. automethod:: rosette(n,angle,axis=2,point=[0.,0.,0.])
      .. automethod:: translatem(None)
      .. automethod:: extrude(n,step=1.,dir=0,autofix=True)
      .. automethod:: divide(div)
      .. automethod:: intersectionWithPlane(p,n)
      .. automethod:: intersectionPointsWithPlane(p,n)
      .. automethod:: intersectionLinesWithPlane(p,n)
      .. automethod:: cutWithPlane(p,n,side='',atol=None,newprops=None)
      .. automethod:: cutAtPlane(p,n,newprops=None,side='+',atol=0.)
      .. automethod:: split(n=1)
      .. automethod:: write(fil,sep=' ',mode='w')
      .. automethod:: read(clas,fil,sep=' ')
      .. automethod:: fromstring(clas,fil,sep=' ',nplex=1,ndim=3,count=1)
      .. automethod:: fromfile(clas,fil,sep=' ',nplex=1)
      .. automethod:: reverseElements()
      .. automethod:: feModel(None)

**Functions defined in the module formex**

   .. autofunction:: vectorLength(vec)
   .. autofunction:: vectorNormalize(vec)
   .. autofunction:: vectorPairAreaNormals(vec1,vec2)
   .. autofunction:: vectorPairArea(vec1,vec2)
   .. autofunction:: vectorPairNormals(vec1,vec2)
   .. autofunction:: vectorTripleProduct(vec1,vec2,vec3)
   .. autofunction:: polygonNormals(x)
   .. autofunction:: pattern(s)
   .. autofunction:: mpattern(s)
   .. autofunction:: intersectionWithPlane(F,p,n)
   .. autofunction:: pointsAt(F,t)
   .. autofunction:: intersectionPointsWithPlane(F,p,n)
   .. autofunction:: intersectionLinesWithPlane(F,p,n,atol=1.e-4)
   .. autofunction:: cut2AtPlane(F,p,n,side='',atol=None,newprops=None)
   .. autofunction:: cut3AtPlane(F,p,n,side='',atol=None,newprops=None)
   .. autofunction:: cutElements3AtPlane(F,p,n,newprops=None,side='',atol=0.)
   .. autofunction:: connect(Flist,nodid=None,bias=None,loop=False)
   .. autofunction:: interpolate(F,G,div,swap=False)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

